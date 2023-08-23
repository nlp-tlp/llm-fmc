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
