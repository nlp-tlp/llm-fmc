import os
import openai
import pandas as pd

test = pd.read_json(
    os.path.join("input_data", "prepared", "test_prepared.jsonl"), lines=True
)
test.head()


# Replace with your model
ft_model = "ada:ft-uwa-system-health-lab-2023-08-16-05-03-47"

print(test["prompt"])
print(test["prompt"][0])

res = openai.Completion.create(
    model=ft_model,
    prompt=test["prompt"][0] + "\n\n###\n\n",
    max_tokens=10,
    temperature=0,
)
print(res)


# results = pd.read_csv("result.csv")
# print(results[results["classification/accuracy"].notnull()].tail(1))
