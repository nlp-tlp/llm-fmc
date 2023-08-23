## ChatGPT-FMC

Experiments in using ChatGPT to perform Failure Mode Classification.

### Commands

First, set your `OPENAI_API_KEY` environment variable in the terminal:

    set OPENAI_API_KEY=<your OpenAI API key>

To install the packages via poetry:

    pip install poetry
    poetry install

Then, run the shell script:

    sh prepare_data.sh

To fine-tune ChatGPT on the data:

    poetry shell
    openai api fine_tunes.create -t "input_data/prepared/train_prepared.jsonl" -v "input_data/prepared/dev_prepared.jsonl" --compute_classification_metrics --classification_n_classes 22 -m ada

Note the `22` is the number of unique classes in your dataset (which is output by `prepare_data.py`).

The model (`-m`) can be `ada`, `davinci`, or any other model listed in the OpenAI docs ([here](https://platform.openai.com/docs/guides/fine-tuning)).

Note also that the number of classes in the training and dev set need to be the same, i.e. every class in the training set also needs to be present in the dev set. I don't know why it is like this but it is a requirement of the fine tuning script.
