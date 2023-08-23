"""Code for preparing the data for openai.
Provides detail to the user on the number of unique classes in their datasets,
which is important as OpenAI won't allow you to fine tune when the train
and dev sets have different numbers of unique classes.
"""
import os
import json
import openai
import time
import sys

openai.api_key = os.getenv("OPENAI_API_KEY")


def finetune_model():
    """Finetune a model in ChatGPT."""

    # Send the train file to OpenAI
    f_train = openai.File.create(
        file=open(os.path.join("input_data", "prepared", "train.jsonl"), "rb"),
        purpose="fine-tune",
    )

    # # Send the dev file to OpenAI
    f_dev = openai.File.create(
        file=open(os.path.join("input_data", "prepared", "dev.jsonl"), "rb"),
        purpose="fine-tune",
    )

    # print(f_train, f_dev)

    # Halt until the files have been processed
    print("Waiting for file processing...", end="")
    while True:
        f_train_status = openai.File.retrieve(f_train["id"])["status"]
        f_dev_status = openai.File.retrieve(f_dev["id"])["status"]

        if f_train_status == "processed" and f_dev_status == "processed":
            break

        # Check every 5 seconds
        time.sleep(5)
        print(".", end="")
        sys.stdout.flush()

    print()
    print("Files ready. Creating fine-tune job...")

    # Create the fine-tuning job
    ftj = openai.FineTuningJob.create(
        training_file="file-ljLATcQWZ99bYEAwKmIsW6Or",
        validation_file="file-83ZKhbwwT6AVUROUBNvb4LNH",
        model="gpt-3.5-turbo",
    )

    ftj_id = ftj["id"]
    print(
        f"Created fine-tune job with id {ftj_id}. Writing to finetune-job.json..."
    )
    with open("finetune-job.json", "w") as f:
        json.dump(ftj, f)

    print("Waiting for fine-tuning to complete...")
    messages = set()

    while True:
        ftj_status = openai.FineTuningJob.retrieve(ftj_id)["status"]
        if ftj_status == "succeeded" or ftj_status == "failed":
            break

        # Print out the messages (fine tuning status)
        # Only print out messages that have not been seen before
        events = openai.FineTuningJob.list_events(ftj_id, limit=10)
        for e in events["data"][::-1]:
            message = e["message"]
            if message not in messages:
                print(message)
                messages.add(message)

        # Only do this every 5 seconds
        time.sleep(5)

    print("Fine tuning complete. Final results:")
    ftj_result = openai.FineTuningJob.retrieve(ftj_id)
    print(ftj_result)

    with open("finetune-result.json", "w") as f:
        json.dump(ftj_result, f)

    # print(openai.FineTuningJob.list(limit=10))

    # openai.FineTuningJob.retrieve("file-mSRymqrmWMVk9aLpRsAUTDPn")


def check_status(ftj_id):
    # print(openai.FineTuningJob.list(limit=10))

    messages = set()
    while True:
        events = openai.FineTuningJob.list_events(ftj_id, limit=10)
        for e in events["data"][::-1]:
            message = e["message"]
            if message not in messages:
                print(message)
                messages.add(message)

        time.sleep(5)

    # print(openai.FineTuningJob.cancel("ftjob-NQoSo5EboTj1S999gESKTrIR"))


def main():
    check_status("ftjob-stYCObzZF0jpR3ZtoCBs7hep")
    # finetune_model()


if __name__ == "__main__":
    main()
