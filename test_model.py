import os
import json
import openai
import numpy as np
from sklearn.metrics import classification_report

# FT_MODEL = "ft:gpt-3.5-turbo-0613:uwa-system-health-lab::7qdnRrbA"
FT_MODEL = "gpt-3.5-turbo"

# EXPERIMENT_NAME = "fine-tuned"
# EXPERIMENT_NAME = "gpt-3.5-turbo"
EXPERIMENT_NAME = "gpt-3.5-turbo-constrained-labels"


def evaluate_model():
    """Evaluate the model by running it on the test dataset.
    First, process the raw test dataset to extract the inputs, and gold outputs.
    Then, run the model on each instance of the test dataset and extract the
    predicted class, appending it to the predicted outputs array.
    Once done, create a classification report using sklearn.

    Raises:
        ValueError: If the raw test dataset does not align with the prepared test
           dataset. This should not happen, and is just a sanity check to avoid
           invalid results.
    """

    test_data = {"inputs": [], "outputs_gold": [], "outputs_pred": []}
    with open(os.path.join("input_data", "raw", "test.txt"), "r") as f:
        for line in f.readlines():
            inp, outp = line.strip().split(",")
            test_data["inputs"].append(inp)
            test_data["outputs_gold"].append(outp)

    # Build an engineered prompt to use in the 'constrained-labels' experiment.
    # This one feeds the list of valid labels to OpenAI prior to classification.
    labels = set()
    for ds in ["train", "dev", "test"]:
        with open(os.path.join("input_data", "raw", f"{ds}.txt"), "r") as f:
            for line in f.readlines():
                inp, outp = line.strip().split(",")
                labels.add(outp)
    label_list = f" Valid failure modes are:\n" + "\n".join(labels)

    constraint = (
        " Your answer should contain only the failure mode and nothing else."
    )

    test_output = []

    with open(os.path.join("input_data", "prepared", "test.jsonl"), "r") as f:
        for i, row in enumerate(f.readlines()):
            row_json = json.loads(row)

            # Sanity check to ensure the raw test data is still the same
            # as the prepared test data
            if (
                row_json["messages"][1]["content"].split("\n\n")[1]
                != test_data["inputs"][i]
            ):
                raise ValueError(
                    f"Test data is not aligned on sentence with id {i}"
                )

            # For gpt-3.5-constrained-labels, append the list of labels at the end
            # of the prompt (so ChatGPT knows which labels are valid).
            #
            if EXPERIMENT_NAME == "gpt-3.5-turbo-constrained-labels":
                row_json["messages"][0]["content"] = (
                    row_json["messages"][0]["content"]
                    + constraint
                    + label_list
                )
            # For regular gpt-3.5, just add the constraint ("Your answer should
            # contain only the failure mode ...") to ensure it does not output
            # spurious details etc.
            elif EXPERIMENT_NAME == "gpt-3.5-turbo":
                row_json["messages"][0]["content"] = (
                    row_json["messages"][0]["content"] + constraint
                )

            if i == 0:
                print("Example prompt: ")
                print(row_json)

            # print(row_json)
            # exit()

            # Create a chat completion for each sentence
            res = openai.ChatCompletion.create(
                model=FT_MODEL, messages=row_json["messages"], temperature=0
            )

            pred = res["choices"][0]["message"]["content"]

            test_output.append((test_data["inputs"][i], pred))

            # Append the predicted class to outputs_pred
            test_data["outputs_pred"].append(pred)
            if i > 0 and i % 5 == 0:
                print(f"Processed {i} rows")

    report = classification_report(
        test_data["outputs_gold"],
        test_data["outputs_pred"],
        labels=np.unique(test_data["outputs_gold"]),
    )

    print(report)

    with open(
        os.path.join("output", f"evaluation_output-{EXPERIMENT_NAME}.txt"), "w"
    ) as f:
        f.write(report)

    with open(
        os.path.join("output", f"model_output-{EXPERIMENT_NAME}.txt"), "w"
    ) as f:
        for row in test_output:
            f.write(f"{row[0]}, {row[1]}\n")

    # for row in test:
    #     print(row)
    # res = openai.ChatCompletion.create(
    #     model=FT_MODEL,
    #     messages=[
    #         to_prompt(row)
    #     ]
    # )
    # print(res)


def main():
    evaluate_model()


if __name__ == "__main__":
    main()


# results = pd.read_csv("result.csv")
# print(results[results["classification/accuracy"].notnull()].tail(1))
