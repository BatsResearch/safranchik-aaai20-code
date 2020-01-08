# Weakly Supervised Sequence Tagging from Noisy Rules - Reproducibility Code

## Getting Started

These instructions will get you a copy of our experiments up and running on your local machine for development and testing purposes.

### Prerequisites

Please clone and download [wiser](https://github.com/BatsResearch/wiser), our framework for training weakly supervising deep sequence taggers for named entity recognition tasks.

To get the code for generative models and other rule-aggregation methods, please download the latest version of [labelmodels](https://https://github.com/BatsResearch/labelmodels), our lightweight implementation of Lightweight of generative label models for weakly supervised machine learning

### Installing

To download the otdependencies required to run our experiments, run

```
pip install -r requirements.txt
```

Then, install SpaCy's small English model by running
'''
python -m spacy download en_core_web_sm
'''


Then, in your virtual enviornment, please run the *labelmodels/setup.py* and *wiser/setup.py* scripts to install the corresponding dependencies.
'''
python setup.py install
'''

## Datasets

Our experiments depend on *six* different datasets that you will need to separately download. Please add all dataset directories to *safranchik-aaai20-code/data* directory.

* BC5CDR:

* [NCBI Disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/): Download and install the complete training, development, and testing sets, and place them in a folder *data/NCBI*.

* LaptopReview:

* English [CoNLL v5](https://catalog.ldc.upenn.edu/LDC2013T19): Download and compile the English dataset version 5.0 , and place it in a folder  *data/conll-formatted-ontonotes-5.0*.

* [Scibert](https://github.com/allenai/scibert): Download the scibert-scivocab-uncased version of the Scibert embeddings, and place the files *weights.tar.gz and *vocab.txt* inside *data/scibert_scibocab_uncased*.

## Authors

* **Esteban Safranchik**
* **Shiying Luo**
* **Stephen H. Bach**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
