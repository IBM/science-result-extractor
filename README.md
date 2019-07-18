# Science-result-extractor

## Introduction 


This repository contains code and two datasets to extract TDMS (Task, Dataset, Metric, Score) tuples from scientific papers in the NLP domain. Please refer to the following paper for the full details:

Yufang Hou, Charles Jochim, Martin Gleize, Francesca Bonin, Debasis Ganguly. Identification of Tasks, Datasets, Evaluation Metrics, and Numeric Scores for Scientific Leaderboards Construction. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), Florence, Italy, 27 July - 2 August 2019

## Extract table content from PDF files

We developed a deterministic PDF table parser based on [GROBID](https://github.com/kermitt2/grobid). To use our parser, follow the steps below:

1) Clone this repository, e.g., 
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


## Read NLP-TDMS and ARC-PDN corpora 

1) Follow the instructions in the [README](data/NLP-TDMS/downloader/README.md) in [data/NLP-TDMS/downloader/](data/NLP-TDMS/downloader/) to download the entire collection of raw PDFs of the NLP-TDMS dataset.  The downloaded PDFs can be moved to [data/NLP-TDMS/pdfFile](./data/NLP-TDMS/pdfFile) (i.e., `mv *.pdf ../pdfFile/.`).

2) For the ARC-PDN corpus, the original pdf files can be downloaded from the [ACL Anthology Reference Corpus (Version 20160301)](https://acl-arc.comp.nus.edu.sg/). We use papers from ACL(P)/EMNLP(D)/NAACL(N) between 2010 and 2015. After uncompressing the downloaded PDF files, put the PDF files into the corresponding directories under the /data/ARC-PDN/ folder, e.g., copy D10 to /data/ARC-PDN/D/D10.

3) We release the parsed NLP-TDMS and ARC-PDN corpora. [NlpTDMSReader](nlpLeaderboard/src/main/java/com/ibm/sre/data/corpus/NlpTDMSReader.java) and [ArcPDNReader](nlpLeaderboard/src/main/java/com/ibm/sre/data/corpus/ArcPDNReader.java) in the corpus package illustrate how to read section and table contents from PDF files in these two corpora. 


## Run experiments based on textual entailment system

We release the training/testing datasets for all experiments described in the paper. You can find them under the data/exp directory. The results reported in the paper are based on the datasets under the [data/exp/few-shot-setup/NLP-TDMS/paperVersion](data/exp/few-shot-setup/NLP-TDMS/paperVersion) directory. We later further clean the datasets (e.g., remove five pdf files from the testing datasets which appear in the training datasets with a different name) and the clean version is under the [data/exp/few-shot-setup/NLP-TDMS](data/exp/few-shot-setup/NLP-TDMS) folder. Below we illustrate how to run experiments on the NLP-TDSM dataset in the few-shot setup to extract TDM pairs. 


1) clone the repository

2) download bert embedding "uncased_L-12_H-768_A-12" and put it under python/bert_sci_te/model

3) on CCC, run subjob_train_exp3.sh and subjob_test_exp3.sh under python/bert_sci_te to train a textual entail model, and test this model on the testing data

4) TEModelEvalOnNLPTDMS in the te package provides methods to evaluate TDMS tuples extraction.

5) GenerateTestDataOnPDFPapers in the te package provides methods to generate testing dataset for any pdf papers.


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
