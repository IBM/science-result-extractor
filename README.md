# Science-result-extractor

## Introduction 

This repository contains code and two datasets to extract TDMS (Task, Dataset, Metric, Score) tuples from scientific papers in the NLP domain.
We envision two primary uses for this repository: (1) to [extract table content from PDF files](#extract-table-content-from-pdf-files), and (2) to [replicate the paper's results or run experiments based on a textual entailment system](#run-experiments-based-on-textual-entailment-system).
Please refer to the following paper for the full details:

Yufang Hou, Charles Jochim, Martin Gleize, Francesca Bonin, Debasis Ganguly. Identification of Tasks, Datasets, Evaluation Metrics, and Numeric Scores for Scientific Leaderboards Construction. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), Florence, Italy, 27 July - 2 August 2019

## Extract table content from PDF files

We developed a deterministic PDF table parser based on [GROBID](https://github.com/kermitt2/grobid). To use our parser, follow the steps below:

1) Fork and clone this repository, e.g., 
```
> git clone https://github.com/IBM/science-result-extractor.git
```

2) Download and install GROBID 0.5.3, following the [installation instructions](https://grobid.readthedocs.io/en/latest/Install-Grobid/#getting-grobid), e.g., 
```
> wget https://github.com/kermitt2/grobid/archive/0.5.3.zip
> unzip 0.5.3.zip
> cd grobid-0.5.3/
> ./gradlew clean install
```
(note that gradlew must be installed beforehand)

3) Configure `pGrobidHome` and `pGrobidProperties` in [config.properties](nlpLeaderboard/config.properties). The default configuration assumes that GROBID directory `grobid-0.5.3` is a sister of the science-result-extractor directory.
```
pGrobidHome=../../grobid-0.5.3/grobid-home
pGrobidProperties=../../grobid-0.5.3/grobid-home/config/grobid.properties 
```

4) [PdfInforExtractor](nlpLeaderboard/src/main/java/com/ibm/sre/pdfparser/PdfInforExtractor.java) provides methods to extract section content and table content from a given PDF file.


## Run experiments based on textual entailment system

We release the training/testing datasets for all experiments described in the paper. You can find them under the data/exp directory. The results reported in the paper are based on the datasets under the [data/exp/few-shot-setup/NLP-TDMS/paperVersion](data/exp/few-shot-setup/NLP-TDMS/paperVersion) directory. We later further clean the datasets (e.g., remove five pdf files from the testing datasets which appear in the training datasets with a different name) and the clean version is under the [data/exp/few-shot-setup/NLP-TDMS](data/exp/few-shot-setup/NLP-TDMS) folder. Below we illustrate how to run experiments on the NLP-TDSM dataset in the few-shot setup to extract TDM pairs. 


1) Fork and clone this repository.

2) Download or clone [BERT](https://github.com/google-research/bert).

3) Copy [run_classifier_sci.py](./bert_tdms/run_classifier_sci.py) into the BERT directory.

3) Download BERT embeddings.  We use the [base uncased models](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip).

4) If we use `BERT_DIR` to point to the directory with the embeddings and `DATA_DIR` to point to the [directory with our train and test data](./data/exp/few-shot-setup/NLP-TDMS/), we can run the textual entailment system with  [run_classifier_sci.py](./bert_tdms/run_classifier_sci.py). For example:

```
> DATA_DIR=../data/exp/few-shot-setup/NLP-TDMS/
> BERT_DIR=./model/uncased_L-12_H-768_A-12/
> python3 run_classifier_sci.py --do_train=true --do_eval=false --do_predict=true --data_dir=${DATA_DIR} --task_name=sci --vocab_file=${BERT_DIR}/vocab.txt --bert_config_file=${BERT_DIR}/bert_config.json --init_checkpoint=${BERT_DIR}/bert_model.ckpt --output_dir=bert_tdms --max_seq_length=512 --train_batch_size=6 --predict_batch_size=6
```

5) [TEModelEvalOnNLPTDMS](nlpLeaderboard/src/main/java/com/ibm/sre/tdmsie/TEModelEvalOnNLPTDMS.java) provides methods to evaluate TDMS tuples extraction.

6) [GenerateTestDataOnPDFPapers](nlpLeaderboard/src/main/java/com/ibm/sre/tdmsie/GenerateTestDataOnPDFPapers.java) provides methods to generate testing dataset for any PDF papers.


### Read NLP-TDMS and ARC-PDN corpora ###

1) Follow the instructions in the [README](data/NLP-TDMS/downloader/README.md) in [data/NLP-TDMS/downloader/](data/NLP-TDMS/downloader/) to download the entire collection of raw PDFs of the NLP-TDMS dataset.  The downloaded PDFs can be moved to [data/NLP-TDMS/pdfFile](./data/NLP-TDMS/pdfFile) (i.e., `mv *.pdf ../pdfFile/.`).

2) For the ARC-PDN corpus, the original pdf files can be downloaded from the [ACL Anthology Reference Corpus (Version 20160301)](https://acl-arc.comp.nus.edu.sg/). We use papers from ACL(P)/EMNLP(D)/NAACL(N) between 2010 and 2015. After uncompressing the downloaded PDF files, put the PDF files into the corresponding directories under the /data/ARC-PDN/ folder, e.g., copy D10 to /data/ARC-PDN/D/D10.

3) We release the parsed NLP-TDMS and ARC-PDN corpora. [NlpTDMSReader](nlpLeaderboard/src/main/java/com/ibm/sre/data/corpus/NlpTDMSReader.java) and [ArcPDNReader](nlpLeaderboard/src/main/java/com/ibm/sre/data/corpus/ArcPDNReader.java) in the corpus package illustrate how to read section and table contents from PDF files in these two corpora. 


## Citing science-result-extractor
Please cite the following paper when using science-result-extractor:

```
@inproceedings{houyufang2019acl,
  title={Identification of Tasks, Datasets, Evaluation Metrics, and Numeric Scores for Scientific Leaderboards Construction},
  author={Hou, Yufang and Jochim, Charles and Gleize, Martin and Bonin, Francesca and Ganguly, Debasis},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, {\em Florence, Italy, 27 July -- 2 August 2019}},
  year      = {2019}
}
```
