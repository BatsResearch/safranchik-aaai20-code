# Weakly Supervised Sequence Tagging from Noisy Rules - Reproducibility Code

## Getting Started

These instructions will get you a copy of our experiments up and running on your local machine for development and testing purposes.

### Installing

In your virtual environment, please install the required dependencies using

```
pip install -r requirements.txt
```
Or alternatively
```
conda install --file requirements.txt
```

## Datasets

Our experiments depend on *six* different datasets that you will need to download separately.

* [BC5CDR](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/): Download and install the train, development, and test BioCreative V CDR corpus data files. Place the three separate files inside data/BC5CD

* [NCBI Disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/): Download and install the complete training, development, and testing sets. Place the three separate files inside *data/NCBI*.

* [LaptopReview](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools): Download the train data V2.0 for the Laptops and Restaurants dataset, and place the *Laptop_Train_v2.xml* file inside *data/LaptopReview*. Then, download the test data - phase B, and place the *Laptops_Test_Data_phaseB.xml* file inside the same directory.

* [CoNLL v5](https://catalog.ldc.upenn.edu/LDC2013T19): Download and compile the English dataset version 5.0, and place it in *data/conll-formatted-ontonotes-5.0*.

* [Scibert](https://github.com/allenai/scibert): Download the scibert-scivocab-uncased version of the Scibert embeddings, and place the files *weights.tar.gz and *vocab.txt* inside *data/scibert_scibocab_uncased*.

* [UMLS](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html): The UMLS dictionaries have been extracted from the UMLS 2018AB dataset and are provided in our code. They are distributed according to the [License Agreement for Use of the UMLS® Metathesaurus®] (https://uts.nlm.nih.gov/help/license/LicenseAgreement.pdf).

* [AutoNER Dictionaries](https://github.com/shangjingbo1226/AutoNER). The AutoNER dictionaries for the BC5CDR, LaptopReview, and NCBI datasets have been  generously provided by Jingbo Shang et al. They have been sourced from the EMNLP 2018 paper "Learning Named Entity Tagger using Domain-Specific Dictionary".

## Citation

Please cite the following paper if you are using our tool. Thank you!

Safranchik Esteban, Shiying Luo, Stephen H. Bach. "Weakly Supervised Sequence Tagging From Noisy Rules". In 34th AAAI Conference on Artificial Intelligence, 2020.

```
@inproceedings{safranchik2020weakly,
  title = {Weakly Supervised Sequence Tagging From Noisy Rules}, 
  author = {Safranchik, Esteban and Luo, Shiying and Bach, Stephen H.}, 
  booktitle = {AAAI}, 
  year = 2020, 
}
```
