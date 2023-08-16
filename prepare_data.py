import os
import pandas as pd
import openai
from sklearn.datasets import fetch_20newsgroups

# categories = ["rec.sport.baseball", "rec.sport.hockey"]
# print("xx")
# sports_dataset = fetch_20newsgroups(
#     subset="train", shuffle=True, random_state=42, categories=categories
# )

# print(sports_dataset["data"][0])

# # Data prep

# labels = [
#     sports_dataset.target_names[x].split(".")[-1]
#     for x in sports_dataset["target"]
# ]
# texts = [text.strip() for text in sports_dataset["data"]]
# df = pd.DataFrame(
#     zip(texts, labels), columns=["prompt", "completion"]
# )  # [:300]
# df.head()

# print("hi")

# df.to_json("sport2.jsonl", orient="records", lines=True)


# Build the map of class -> id_class
# Such as "leak" -> "1_leak"
# (OpenAI requires the classes to each start with a unique token)
class_map = {}
for ds in ["train", "dev", "test"]:
    unique_classes = set()
    fn = os.path.join("input_data", "raw", f"{ds}.txt")
    with open(fn, "r") as f:
        data = [line.strip().split(",") for line in f.readlines()]
    for text, label in data:
        parsed_label = label.replace(" ", "_")
        if label not in class_map:
            class_map[label] = f"{len(class_map)}_{parsed_label}"

for ds in ["train", "dev", "test"]:
    unique_classes = set()
    fn = os.path.join("input_data", "raw", f"{ds}.txt")
    with open(fn, "r") as f:
        data = [line.strip().split(",") for line in f.readlines()]
    for i, (text, label) in enumerate(data):
        data[i][1] = class_map[label]

    df = pd.DataFrame(data, columns=["prompt", "completion"])
    df.head()
    df.to_json(
        os.path.join("input_data", "intermediate", f"{ds}.jsonl"),
        orient="records",
        lines=True,
    )

    print(f"There are {len(unique_classes)} unique classes in the {ds} set.")
    with open(
        os.path.join("input_data", "intermediate", f"classes_{ds}.txt"), "w"
    ) as f:
        f.write("\n".join(sorted(list(unique_classes))))

print(class_map)
