## Fine-tuning Classification with BERT

We fine-tune BERT for our natural language inference (NLI) task, predicting task-dataset-metric (TDM) triples.

To run the code, download or clone a copy of [BERT](https://github.com/google-research/bert) and copy the [run\_classifier\_sci.py](./run_classifier_sci.py) script into the BERT directory.
This script adds a new `DataProcessor`, "SciProcessor", for training and testing on our TDM data.

Example usage:

```
> DATA_DIR=../data/exp/few-shot-setup/NLP-TDMS/
> BERT_DIR=./model/uncased_L-12_H-768_A-12/
> python3 run_classifier_sci.py --do_train=true --do_eval=false --do_predict=true --data_dir=${DATA_DIR} --task_name=sci --vocab_file=${BERT_DIR}/vocab.txt --bert_config_file=${BERT_DIR}/bert_config.json --init_checkpoint=${BERT_DIR}/bert_model.ckpt --output_dir=/tmp/ --max_seq_length=512 --train_batch_size=6 --predict_batch_size=6
```

The `DATA_DIR` should point to the training and testing data in [this repository](../data/) and the `BERT_DIR` should have pre-trained BERT models (we use the [base uncased models](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)). 

See the repository's [README.md](../README.md) for instructions on how to produce and training data and evaluate results.
