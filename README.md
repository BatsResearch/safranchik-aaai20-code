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

* [UMLS](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html): Download the 2018AB version of knwoeldge sources, and place them a folder called *data/2018AB*. Then, extract the *umls_antibiotic.txt*, *umls_body_part.txt*, *umls_disease_or_syndrome.txt*, *umls_element_ion_or_isotope.txt*, and *umls_organic_chemical.txt* files and place them in *data/umls*.

* [AutoNER Dictionaries](). Please download the *dict_core.txt* and *dict_full.txt* files for the BC5CDR, NCBI-Disease, and LaptopReview dataset. Then, place them into the folders *data/AutoNER_dicts/BC5CDR*, *data/AutoNER_dicts/NCBI-Disease*, and *data/AutoNER_dicts/LaptopReview* respectively.


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
