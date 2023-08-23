import os
import json
import openai
from sklearn.metrics import classification_report

FT_MODEL = "ft:gpt-3.5-turbo-0613:uwa-system-health-lab::7qdnRrbA"


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

            # Create a chat completion for each sentence
            res = openai.ChatCompletion.create(
                model=FT_MODEL, messages=row_json["messages"], temperature=0
            )

            # Append the predicted class to outputs_pred
            test_data["outputs_pred"].append(
                res["choices"][0]["message"]["content"]
            )
            if i % 5 == 0:
                print(f"Processed {i} of {len(row_json['messages'])} rows")

    report = classification_report(
        test_data["outputs_gold"], test_data["outputs_pred"]
    )

    print(report)

    with open(os.path.join("output", "evaluation_output.txt"), "w") as f:
        f.write(report)

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
