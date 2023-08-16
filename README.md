## ChatGPT-FMC

Experiments in using ChatGPT to perform Failure Mode Classification.


### Commands

First, set your `OPENAI_API_KEY` environment variable in the terminal:

    set OPENAI_API_KEY=<your OpenAI API key>

To install the packages via poetry:

    pip install poetry
    poetry install

To prepare the data (get it into the prompt format):

    poetry shell
    python prepare_data.py

This will read in the data from `input_data/raw` and process it into a JSONL file.

To clean it up to make it ready for fine-tuning ChatGPT (The responses are Y, Y, n, Y, i.e. yes to everything aside from splitting it into train and dev):

    openai tools fine_tunes.prepare_data -f input_data/intermediate/train.jsonl
    openai tools fine_tunes.prepare_data -f input_data/intermediate/dev.jsonl
    openai tools fine_tunes.prepare_data -f input_data/intermediate/test.jsonl

Then, copy `train_prepared.jsonl`, `dev_prepared.jsonl` and `test_prepared.jsonl` out of `input_data/intermediate` and into `input_data.prepared`. A shell command to do this:

    mv input_data/intermediate/train_prepared.jsonl input_data/prepared
    mv input_data/intermediate/dev_prepared.jsonl input_data/prepared
    mv input_data/intermediate/test_prepared.jsonl input_data/prepared

To fine-tune ChatGPT on the data:

    openai api fine_tunes.create -t "input_data/prepared/train_prepared.jsonl" -v "input_data/prepared/dev_prepared.jsonl" --compute_classification_metrics --classification_n_classes 22 -m ada

Note the `22` is the number of unique classes in your dataset (which is output by `prepare_data.py`).

Note also that the number of classes in the training and dev set need to be the same, i.e. every class in the training set also needs to be present in the dev set. I don't know why it is like this but it is a requirement of the fine tuning script.