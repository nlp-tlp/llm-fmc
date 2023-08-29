## ChatGPT-FMC

Experiments in using ChatGPT to perform Failure Mode Classification.

### Commands

First, set your `OPENAI_API_KEY` environment variable in the terminal:

    set OPENAI_API_KEY=<your OpenAI API key>

To install the packages via poetry:

    pip install poetry
    poetry install

Before running anything, run the poetry shell:

    poetry shell

Then, run the script to prepare the data:

    python prepare_data.py

You might like to use OpenAI's 'file checking' script to ensure your data is suitable for feeding into the fine tuner:

    python check_file.py

To fine-tune ChatGPT on the data:

    python finetune_model.py

To test the model, open `test_model.py` and change the `FT_MODEL` variable to the fine tuned model from the previous step. Then, run:

    python test_model.py

A classification report will be printed and results will be saved into the `output` folder.

### Experiments

#### Fine-tuned

The GPT-3.5 model has been fine tuned on our dataset of 500 observation -> failure mode pairs. The prompt being fed into the model in `test_model.py` is:

    Determine the failure mode of the following sentence:

    smoking up

The model will predict the exact failure mode:

    Overheating

#### GPT-3.5

As above, but with no fine tuning. We adjust the prompt to encourage the model to only output the failure mode, and no other text (such as "The failure mode is ..."):

    Determine the failure mode of the following sentence:

    smoking up

    Your answer should contain only the failure mode and nothing else.

### GPT-3.5-Constrained-Labels

As above, but the prompt is further engineered to elicit a certain response from the model. We explicitly provide the list of labels that the model can select from. The prompt looks as follows:

    Determine the failure mode of the following sentence:

    jammed

    Your answer should contain only the failure mode and nothing else. Valid failure modes are:
    Leaking
    Structural deficiency
    Other
    Low output
    Spurious stop
    Contamination
    Erratic output
    Electrical
    Failure to start on demand
    Vibration
    Fail to open
    Fail to close
    Noise
    Plugged / choked
    Overheating
    Breakdown
    Failure to stop on demand
    Failure to rotate
    High output
    Abnormal instrument reading
    Fail to function
    Minor in-service problems
