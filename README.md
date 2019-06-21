# Science-result-extractor

## Introduction 
test

This repository contains code and two datasets to extract TDMS (Task, Dataset, Metric, Score) tuples from scientific papers in the NLP domain. Please refer to the following paper for the full details:

Yufang Hou, Charles Jochim, Martin Gleize, Francesca Bonin, Debasis Ganguly. Identification of Tasks, Datasets, Evaluation Metrics, and Numeric Scores for Scientific Leaderboards Construction. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), Florence, Italy, 27 July - 2 August 2019

## Extract table content from PDF files

We developed a derterministic PDF table parser based on Grobid. To use our parser, following the steps below:

1) clone the repository
2) following the instructions in https://grobid.readthedocs.io/en/latest/Install-Grobid/#getting-grobid to install grobid 0.5.3, the following steps works for me (please refer to the link for installation problems):

> wget https://github.com/kermitt2/grobid/archive/0.5.3.zip

> unzip 0.5.3.zip

> ./gradlew clean install

3) config pGrobidHome and pGrobidProperties in the config.properties file. For example, mine looks like the following:

pGrobidHome=/Users/yhou/git/grobid-0.5.3/grobid-home

pGrobidProperties=/Users/yhou/git/grobid-0.5.3/grobid-home/config/grobid.properties 

4) PdfInfoExtractor in the pdfparser package provides methods to extract section content and table content from a given PDF file.


## Read NLP-TDMS and ARC-PDN corpora 

1) clone the repository 

2) following the readme file in the data/NLP-TDMS/downloader/ to download the entire collection of raw PDFs of the NLP-TDMS dataset, put them in the /data/NLP-TDMS/pdfFile directory.

3) for the ARC-PDN corpus, the original pdf files can be downloaded from ACL Anthology Reference Corpus (Version 20160301) (https://acl-arc.comp.nus.edu.sg/). We use papers from ACL(P)/EMNLP(D)/NAACL(N) between 2010 and 2015. After uncompressed the downloaded PDF files, put the PDF files into the corresponding directories under the /data/ARC-PDN/ folder, e.g., copy D10 to /data/ARC-PDN/D/D10

4) We release the pre-parsed NLP-TDMS and ARC-PDN corpora. NlpTDMSReader and ArcPDNReader in the corpus package illustrates how to read section and table contents from pdf files in these two corpora. 


## Run exepriments based on textual entailment system

We release the training/testing datasets for all experiments described in the paper. You can find them under the data/exp directory. The results reported in the paper is based on the datasets under the data/exp/few-shot-setup/NLP-TDMS/paperVersion directory. We later further clean the datasets (e.g., remove five pdf files from the testing datasets which appear in the training datasets with a different name) and the clean version is under the data/exp/few-shot-setup/NLP-TDMS folder. Below we illustrates how to run experiments on the NLP-TDSM dataset in the few-shot setup to extract TDM pairs. 


1) clone the repository

2) download bert embedding "uncased_L-12_H-768_A-12" and put it under python/bert_sci_te/model

3) on CCC, run subjob_train_exp3.sh and subjob_test_exp3.sh under python/bert_sci_te to train a textual entail model, and test this model on the testing data

4) TEModelEvalOnNLPTDMS in the te package provides methods to evaluate TDMS tuples extraction.

5) GenerateTestDataOnPDFPapers in the te package provides methods to generate testing dataset for any pdf papers.
