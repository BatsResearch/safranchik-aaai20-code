#!/usr/bin/env python
# coding: utf-8

# In[27]:

from wiser.data.dataset_readers.laptops import LaptopsDatasetReader
from wiser.lf import LabelingFunction, LinkingFunction, DictionaryMatcher
from wiser.generative import get_label_to_ix, get_rules
from labelmodels import *
from wiser.generative import train_generative_model
from labelmodels import LearningConfig
from wiser.generative import evaluate_generative_model
from wiser.data import save_label_distribution
from wiser.eval import *
import pickle

# Loads Data

# Loads Laptops Data

root = "../data/"
reader = LaptopsDatasetReader()
train_data = reader.read(root + 'LaptopReview/Laptop_Train_v2.xml')
test_data = reader.read(root + 'LaptopReview/Laptops_Test_Gold.xml')


laptops_docs = train_data + test_data


# Selects a random 20% of train_data to use as a dev set
import numpy as np
np.random.seed(0)
np.random.shuffle(train_data)
cutoff = int(len(train_data)/5)
dev_data = train_data[:cutoff]
train_data = train_data[cutoff:]


# Loads Dictionaries

dict_core = set()
with open(root + 'autoner_dicts/LaptopReview/dict_core.txt') as f:
    for line in f.readlines():
        line = line.strip().split()
        term = tuple(line[1:])
        dict_core.add(term)


dict_full = set()

with open(root + 'autoner_dicts/LaptopReview/dict_full.txt') as f:
    for line in f.readlines():
        line = line.strip().split()
        if len(line) > 1:
            dict_full.add(tuple(line))


# Applies Labeling Functions

lf = DictionaryMatcher("CoreDictionary", dict_core, uncased=True, i_label="I")
lf.apply(laptops_docs)


other_terms = [['BIOS'], ['color'], ['cord'], ['hinge'], ['hinges'],
               ['port'], ['speaker']]
lf = DictionaryMatcher("OtherTerms", other_terms, uncased=True, i_label="I")
lf.apply(laptops_docs)


class ReplaceThe(LabelingFunction):
    def apply_instance(self, instance):
        tokens = [token.text for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        for i in range(len(tokens) - 2):
            if tokens[i].lower() == 'replace' and tokens[i+1].lower() == 'the':
                if instance['tokens'][i+2].pos_ == "NOUN":
                    labels[i] = 'O'
                    labels[i+1] = 'O'
                    labels[i+2] = 'I'

        return labels

lf = ReplaceThe()
lf.apply(laptops_docs)


class iStuff(LabelingFunction):
    def apply_instance(self, instance):
        tokens = [token.text for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        for i in range(len(tokens)):
            if len(tokens[i]) > 1 and tokens[i][0] == 'i' and tokens[i][1].isupper():
                labels[i] = 'I'

        return labels

lf = iStuff()
lf.apply(laptops_docs)


class Feelings(LabelingFunction):
    feeling_words = {"like", "liked", "love", "dislike", "hate"}

    def apply_instance(self, instance):
        tokens = [token.text for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        for i in range(len(tokens) - 2):
            if tokens[i].lower() in self.feeling_words and tokens[i+1].lower() == 'the':
                if instance['tokens'][i+2].pos_ == "NOUN":
                    labels[i] = 'O'
                    labels[i+1] = 'O'
                    labels[i+2] = 'I'

        return labels

lf = Feelings()
lf.apply(laptops_docs)


class ProblemWithThe(LabelingFunction):
    def apply_instance(self, instance):
        tokens = [token.text for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        for i in range(len(tokens) - 3):
            if tokens[i].lower() == 'problem' and tokens[i+1].lower() == 'with' and tokens[i+2].lower() == 'the':
                if instance['tokens'][i+3].pos_ == "NOUN":
                    labels[i] = 'O'
                    labels[i+1] = 'O'
                    labels[i+2] = 'O'
                    labels[i+3] = 'I'

        return labels

lf = ProblemWithThe()
lf.apply(laptops_docs)


class External(LabelingFunction):
    def apply_instance(self, instance):
        tokens = [token.text for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        for i in range(len(tokens) - 1):
            if tokens[i].lower() == 'external':
                    labels[i] = 'I'
                    labels[i+1] = 'I'

        return labels

lf = External()
lf.apply(laptops_docs)


stop_words = {"a", "and", "as", "be", "but", "do", "even",
              "for", "from",
              "had", "has", "have", "i", "in", "is", "its", "just",
              "my", "no", "not", "of", "on", "or",
              "that", "the", "these", "this", "those", "to", "very",
              "what", "which", "who", "with"}

class StopWords(LabelingFunction):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].lemma_ in stop_words:
                labels[i] = 'O'
        return labels

lf = StopWords()
lf.apply(laptops_docs)


class Punctuation(LabelingFunction):
    pos = {"PUNCT"}
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i, pos in enumerate([token.pos_ for token in instance['tokens']]):
            if pos in self.pos:
                labels[i] = 'O'

        return labels

lf = Punctuation()
lf.apply(laptops_docs)


class Pronouns(LabelingFunction):
    pos = {"PRON"}
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i, pos in enumerate([token.pos_ for token in instance['tokens']]):
            if pos in self.pos:
                labels[i] = 'O'

        return labels

lf = Pronouns()
lf.apply(laptops_docs)


class NotFeatures(LabelingFunction):
    keywords = {"laptop", "computer", "pc"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].lemma_ in self.keywords:
                labels[i] = 'O'
        return labels

lf = NotFeatures()
lf.apply(laptops_docs)


class Adv(LabelingFunction):
    pos = {"ADV"}
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i, pos in enumerate([token.pos_ for token in instance['tokens']]):
            if pos in self.pos:
                labels[i] = 'O'

        return labels

lf = Adv()
lf.apply(laptops_docs)


# Applies Linking Functions

class CompoundPhrase(LinkingFunction):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if instance['tokens'][i-1].dep_ == "compound":
                links[i] = 1

        return links

lf = CompoundPhrase()
lf.apply(laptops_docs)


from wiser.lf import ElmoLinkingFunction

lf = ElmoLinkingFunction(.8)
lf.apply(laptops_docs)


class ExtractedPhrase(LinkingFunction):
    def __init__(self, terms):
        self.term_dict = {}

        for term in terms:
            term = [token.lower() for token in term]
            if term[0] not in self.term_dict:
                self.term_dict[term[0]] = []
            self.term_dict[term[0]].append(term)

        # Sorts the terms in decreasing order so that we match the longest first
        for first_token in self.term_dict.keys():
            to_sort = self.term_dict[first_token]
            self.term_dict[first_token] = sorted(
                to_sort, reverse=True, key=lambda x: len(x))


    def apply_instance(self, instance):
        tokens = [token.text.lower() for token in instance['tokens']]
        links = [0] * len(instance['tokens'])

        i = 0
        while i < len(tokens):
            if tokens[i] in self.term_dict:
                candidates = self.term_dict[tokens[i]]
                for c in candidates:
                    # Checks whether normalized AllenNLP tokens equal the list
                    # of string tokens defining the term in the dictionary
                    if i + len(c) <= len(tokens):
                        equal = True
                        for j in range(len(c)):
                            if tokens[i + j] != c[j]:
                                equal = False
                                break

                        # If tokens match, labels the instance tokens
                        if equal:
                            for j in range(i + 1, i + len(c)):
                                links[j] = 1
                            i = i + len(c) - 1
                            break
            i += 1

        return links

lf = ExtractedPhrase(dict_full)
lf.apply(laptops_docs)


class ConsecutiveCapitals(LinkingFunction):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        # We skip the first pair since the first
        # token is almost always capitalized
        for i in range(2, len(instance['tokens'])):
            # We skip this token if it all capitals
            all_caps = True
            text = instance['tokens'][i].text
            for char in text:
                if char.islower():
                    all_caps = False
                    break

            if not all_caps and text[0].isupper()             and instance['tokens'][i-1].text[0].isupper():
                links[i] = 1

        return links

lf = ConsecutiveCapitals()
lf.apply(laptops_docs)


print(score_labels_majority_vote(test_data, span_level=True))
print('--------------------')

save_label_distribution('output/dev_data.p', dev_data)
save_label_distribution('output/test_data.p', test_data)

gen_label_to_ix, disc_label_to_ix = get_label_to_ix(train_data)

dist = get_mv_label_distribution(train_data, disc_label_to_ix, 'O')
save_label_distribution('output/train_data_mv.p', train_data, dist)
dist = get_unweighted_label_distribution(train_data, disc_label_to_ix, 'O')
save_label_distribution('output/train_data_unweighted.p', train_data, dist)

epochs = 5


""" Naive Bayes Model"""
# Defines the model
gen_label_to_ix, disc_label_to_ix = get_label_to_ix(train_data)
tagging_rules, linking_rules = get_rules(train_data)
nb = NaiveBayes(len(gen_label_to_ix)-1, len(tagging_rules), init_acc=0.9, acc_prior=0.01, balance_prior=1.0)

# Trains the model
p, r, f1 = train_generative_model(nb, train_data, dev_data, epochs, gen_label_to_ix, LearningConfig())

# Evaluates the model
print('NB: \n' + str(evaluate_generative_model(model=nb, data=test_data, label_to_ix=gen_label_to_ix)))
print('--------------------')

# Saves the model
label_votes, link_votes, seq_starts = get_generative_model_inputs(train_data, gen_label_to_ix)
p_unary = nb.get_label_distribution(label_votes)
save_label_distribution('output/train_data_nb.p', train_data, p_unary, None, gen_label_to_ix, disc_label_to_ix)


""" HMM Model"""
# Defines the model
gen_label_to_ix, disc_label_to_ix = get_label_to_ix(train_data)
tagging_rules, linking_rules = get_rules(train_data)
hmm = HMM(len(gen_label_to_ix)-1, len(tagging_rules), init_acc=0.9, acc_prior=1, balance_prior=10)

# Trains the model
p, r, f1 = train_generative_model(hmm, train_data, dev_data, epochs, label_to_ix=gen_label_to_ix, config=LearningConfig())

# Evaluates the model
print('HMM: \n' + str(evaluate_generative_model(model=hmm, data=test_data, label_to_ix=gen_label_to_ix)))
print('--------------------')

# Saves the model
label_votes, link_votes, seq_starts = get_generative_model_inputs(train_data, gen_label_to_ix)
p_unary, p_pairwise = hmm.get_label_distribution(label_votes, seq_starts)
save_label_distribution('output/train_data_hmm.p', train_data, p_unary, p_pairwise, gen_label_to_ix, disc_label_to_ix)


""" Linked HMM Model """
# Defines the model
gen_label_to_ix, disc_label_to_ix = get_label_to_ix(train_data)
tagging_rules, linking_rules = get_rules(train_data)
link_hmm = LinkedHMM(num_classes=len(gen_label_to_ix)-1, num_labeling_funcs=len(tagging_rules),
                num_linking_funcs=len(linking_rules), init_acc=0.9, acc_prior=1, balance_prior=10)

# Trains the model
p, r, f1 = train_generative_model(link_hmm, train_data, dev_data, epochs, label_to_ix=gen_label_to_ix, config=LearningConfig())

# Evaluates the model
print('Linked HMM: \n' + str(evaluate_generative_model(model=link_hmm, data=test_data, label_to_ix=gen_label_to_ix)))

# Saves the model
inputs = get_generative_model_inputs(train_data, gen_label_to_ix)
p_unary, p_pairwise = link_hmm.get_label_distribution(*inputs)
save_label_distribution('output/train_data_link_hmm.p', train_data, p_unary, p_pairwise, gen_label_to_ix, disc_label_to_ix)
