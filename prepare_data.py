"""Code for preparing the data for openai.
Provides detail to the user on the number of unique classes in their datasets,
which is important as OpenAI won't allow you to fine tune when the train
and dev sets have different numbers of unique classes.
"""
import os
import json

# Maximum number of rows to include in the prepared dataset.
# Set to a low number just to test out the fine-tuning etc.
MAX_ROWS = 10000

# Message that is included with every prompt.
SYSTEM_CONTENT = (
    "You are a chatbot that classifies the user's query into a failure mode."
)


def prepare_data():
    """Iterate over each dataset (train, dev and test), converting
    them into Pandas dataframes. Save them as a JSONL file where each
    line is a prompt: completion pair.

    Args:
        label_map (dict): The mapping between labels and their ids.
    """

    for ds in ["train", "dev", "test"]:
        dataset = []

        fn = os.path.join("input_data", "raw", f"{ds}.txt")
        with open(fn, "r") as f:
            data = [line.strip().split(",") for line in f.readlines()]
        for i, (text, label) in enumerate(data[:MAX_ROWS]):
            messages = []
            messages.append({"role": "system", "content": SYSTEM_CONTENT})
            messages.append(
                {
                    "role": "user",
                    "content": "Determine the failure mode of "
                    f"the following sentence:\n\n{text}",
                }
            )
            messages.append({"role": "assistant", "content": label})
            dataset.append({"messages": messages})

        with open(
            os.path.join("input_data", "prepared", f"{ds}.jsonl"), "w"
        ) as f:
            for row in dataset:
                f.write(json.dumps(row))
                f.write("\n")


def main():
    prepare_data()


if __name__ == "__main__":
    main()
