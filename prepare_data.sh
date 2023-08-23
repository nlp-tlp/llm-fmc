# To prepare the data (get it into the prompt format):
# This will read in the data from `input_data/raw` and process it into a JSONL file.

python -m poetry run python prepare_data.py

#To clean it up to make it ready for fine-tuning ChatGPT
# (The responses are Y, Y, n, Y, i.e. yes to everything aside
# from splitting it into train and dev):

python -m poetry run openai tools fine_tunes.prepare_data -f ./input_data/intermediate/train.jsonl < res.txt
python -m poetry run openai tools fine_tunes.prepare_data -f ./input_data/intermediate/dev.jsonl < res.txt
python -m poetry run openai tools fine_tunes.prepare_data -f ./input_data/intermediate/test.jsonl < res.txt

# # # Then, copy `train_prepared.jsonl`, `dev_prepared.jsonl` and `test_prepared.jsonl`
# # # out of `input_data/intermediate` and into `input_data.prepared`:

mv ./input_data/intermediate/train_prepared.jsonl ./input_data/prepared
mv ./input_data/intermediate/dev_prepared.jsonl ./input_data/prepared
mv ./input_data/intermediate/test_prepared.jsonl ./input_data/prepared

