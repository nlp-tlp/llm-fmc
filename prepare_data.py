"""Code for preparing the data for openai.
Provides detail to the user on the number of unique classes in their datasets,
which is important as OpenAI won't allow you to fine tune when the train
and dev sets have different numbers of unique classes.
"""
import os
import pandas as pd


def build_label_map():
    """
    Build the map of class -> id
    Such as "leak" -> "1"
    (OpenAI requires the classes to each start with a unique token)

    Returns:
        dict: The label map, as above.
    """
    label_map = {}
    unique_labels = {"train": set(), "dev": set(), "test": set()}
    for ds in ["train", "dev", "test"]:
        fn = os.path.join("input_data", "raw", f"{ds}.txt")
        with open(fn, "r") as f:
            data = [line.strip().split(",") for line in f.readlines()]
        for text, label in data:
            if label not in label_map:
                label_id = len(label_map)
                label_map[label] = label_id
            unique_labels[ds].add(label)

    print(
        f"There are {len(label_map.keys())} unique classes across the train, "
        "dev and test sets."
    )
    if unique_labels["train"] != unique_labels["dev"]:
        print(
            "Warning: some labels do not appear in both the train and dev set:"
        )
        print(unique_labels["train"].difference(unique_labels["dev"]))

    # Write the label map to disk
    with open(
        os.path.join("input_data", "intermediate", f"label_map.txt"),
        "w",
    ) as f:
        for label in sorted(list(label_map), key=label_map.get):
            label_id = label_map[label]
            f.write(f"{label_id} {label}\n")

    return label_map


def prepare_data(label_map: dict):
    """Iterate over each dataset (train, dev and test), converting
    them into Pandas dataframes. Save them as a JSONL file where each
    line is a prompt: completion pair.

    Args:
        label_map (dict): The mapping between labels and their ids.
    """
    for ds in ["train", "dev", "test"]:
        fn = os.path.join("input_data", "raw", f"{ds}.txt")
        with open(fn, "r") as f:
            data = [line.strip().split(",") for line in f.readlines()]
        for i, (text, label) in enumerate(data):
            data[i][1] = label_map[label]

        df = pd.DataFrame(data, columns=["prompt", "completion"])
        df.head()
        df.to_json(
            os.path.join("input_data", "intermediate", f"{ds}.jsonl"),
            orient="records",
            lines=True,
        )


def main():
    label_map = build_label_map()
    prepare_data(label_map)


if __name__ == "__main__":
    main()
