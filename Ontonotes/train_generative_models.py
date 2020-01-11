from wiser.data.dataset_readers import SrlReaderIOB1
from wiser.lf import TaggingRule, LinkingRule
from allennlp.data.fields import MetadataField
from wiser.generative import get_label_to_ix, get_rules
from labelmodels import *
from wiser.generative import train_generative_model
from labelmodels import LearningConfig
from wiser.generative import evaluate_generative_model
from wiser.data import save_label_distribution
from wiser.eval import *
from collections import Counter
from tqdm import tqdm
import spacy
from util import *
import numpy as np
root_directory = '../'

reader = SrlReaderIOB1(used_tags={'I-ARG0', 'I-ARG1', 'I-ARGM-NEG', 'O'})
train_data = reader.read('../data/conll-formatted-ontonotes-5.0/data/train')
dev_data = reader.read(
    '../data/conll-formatted-ontonotes-5.0/data/development')
test_data = reader.read('../data/conll-formatted-ontonotes-5.0/data/test')
ontonotes_docs = train_data + test_data + dev_data


verbs = {'eat', 'ate', 'love', 'call', 'walked'}

reduced_train = []
reduced_test = []
reduced_dev = []

counter = Counter()
for ix, dataset in enumerate([train_data, test_data, dev_data]):

    if ix == 0:
        reduced_dataset = reduced_train
    elif ix == 1:
        reduced_dataset = reduced_test
    else:
        reduced_dataset = reduced_dev

    for instance in dataset:
        verb = instance['metadata'].metadata['verb']

        if verb in verbs:
            reduced_dataset.append(instance)

        if verb is not None:
            counter[verb] += 1

reduced_ontonotes = reduced_train + reduced_dev + reduced_test

nlp = spacy.load('en_core_web_sm')
cnt = Counter()
noun_phrases = {}
preposition_phrases = {}

for verb in verbs:
    noun_phrases[verb] = set()

for instance in tqdm(reduced_ontonotes):

    doc = spacy.tokens.doc.Doc(nlp.vocab,
                               words=[t.text for t in instance['tokens']][1:])

    for name, proc in nlp.pipeline:
        doc = proc(doc)

    instance.add_field('dependency', MetadataField(doc))

    for tag in [t for t in instance['tags']]:
        cnt[tag] += 1

    tokens = [t.text for t in instance['tokens']]
    tok = [t for t in instance['tokens']]
    verb_index = np.where(
        np.array([i for i in instance['verb_indicator']]) == 1)[0][0]
    verb = tokens[verb_index]

    doc = nlp(' '.join(tokens))

    for chunk in doc.noun_chunks:
        phrase = [str(w) for w in chunk]
        while phrase[0] in {"'", ".", ",", '""', ':'}:
            del phrase[0]

        while phrase[-1] in {"'", ".", ",", '""', ':'}:
            del phrase[-1]

        noun_phrases[verb].add(tuple(phrase))


class Verb(TaggingRule):

    def apply_instance(self, instance):

        labels = ['ABS'] * len(instance['tokens'])

        if sum(instance['verb_indicator']) != 1:
            return labels

        verb_index = np.where(
            np.array([i for i in instance['verb_indicator']]) == 1)[0][0]
        labels[verb_index] = 'O'
        labels[0] = 'O'
        return labels


lf = Verb()
lf.apply(reduced_ontonotes)


class Negators(TaggingRule):
    def apply_instance(self, instance):

        labels = np.full((len(instance['tokens'])), 'ABS', dtype=object)

        verb_index = instance['verb_index'].sequence_index
        verb = [t for t in instance['dependency'].metadata][verb_index]
        indices = [token.i + 1 for token in get_tree_negators(verb)]

        labels[indices] = ['I-ARGM-NEG'] * len(indices)

        return labels


lf = Negators()
lf.apply(reduced_ontonotes)


class Subject(TaggingRule):

    def apply_instance(self, instance):

        labels = np.full((len(instance['tokens'])), 'ABS', dtype=object)

        verb_index = instance['verb_index'].sequence_index
        verb = instance['dependency'].metadata[verb_index]

        indices = [token.i + 1 for token in get_subject(verb)]
        labels[indices] = ['I-ARG0'] * len(indices)

        return labels


lf = Subject()
lf.apply(reduced_ontonotes)


class Object(TaggingRule):

    def apply_instance(self, instance):

        labels = np.full((len(instance['tokens'])), 'ABS', dtype=object)

        verb_index = instance['verb_index'].sequence_index
        verb = [t for t in instance['dependency'].metadata][verb_index]

        indices = [token.i + 1 for token in get_object(verb)]
        labels[indices] = ['I-ARG1'] * len(indices)

        return labels


lf = Object()
lf.apply(reduced_ontonotes)


class Modifiers(TaggingRule):

    def apply_instance(self, instance):

        labels = np.full((len(instance['tokens'])), 'ABS', dtype=object)

        verb_index = instance['verb_index'].sequence_index
        verb = [t for t in instance['dependency'].metadata][verb_index]

        indices = [token.i + 1 for token in get_tree_modifiers(verb)]
        labels[indices] = ['O'] * len(indices)

        return labels


lf = Modifiers()
lf.apply(reduced_ontonotes)


other_punc = {"?", "!", ";", ":",
              "%", "<", ">", "=",
              "\\", ".", r"\.", r"\!",
              r"\?", ",", "-", '#',
              "''", "'", '""', '"'}


class Punctuation(TaggingRule):

    def apply_instance(self, instance):

        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].text in other_punc:
                labels[i] = 'O'
        return labels


lf = Punctuation()
lf.apply(reduced_ontonotes)


# Run after Subject LF
class BeforeSubject(TaggingRule):

    def apply_instance(self, instance):

        labels = ['ABS'] * len(instance['tokens'])
        tokens = [t for t in instance['tokens']]
        token_dependencies = [t for t in instance['dependency'].metadata]

        subjects = np.where(
            np.array([t for t in instance['WISER_LABELS']['Subject']]) == 'I-ARG0')[0]

        if len(subjects) == 0:
            return labels

        first_subject_index = min(subjects)
        for i in range(first_subject_index - 1):
            if token_dependencies[i].pos_ in {'NOUN', 'VERB', 'PUNCT'}:
                labels[i + 1] = 'O'

        return labels


lf = BeforeSubject()
lf.apply(reduced_ontonotes)


# Run after Object LF
class AfterObject(TaggingRule):

    def apply_instance(self, instance):

        labels = ['ABS'] * len(instance['tokens'])
        tokens = [t for t in instance['tokens']]
        token_dependencies = [t for t in instance['dependency'].metadata]

        objects = np.where(
            np.array([t for t in instance['WISER_LABELS']['Object']]) == 'I-ARG1')[0]

        if len(objects) == 0:
            return labels

        last_object_index = max(objects)
        for i in range(last_object_index, len(token_dependencies) - 1):
            if token_dependencies[i].pos_ in {
                    'NOUN', 'VERB', 'PUNCT'} and i + 1 < len(labels):
                labels[i + 1] = 'O'

        return labels


lf = AfterObject()
lf.apply(reduced_ontonotes)


class PossessivePhrase(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if instance['tokens'][i -
                                  1].text == "'s" or instance['tokens'][i].text == "'s":
                links[i] = 1

        return links


lf = PossessivePhrase()
lf.apply(reduced_ontonotes)


class HyphenatedPhrase(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens']) - 1):
            if instance['tokens'][i].text == "-":
                links[i] = 1
                links[i + 1] = 1

        return links


lf = HyphenatedPhrase()
lf.apply(reduced_ontonotes)


class PostHyphen(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if instance['tokens'][i - 1].text == "-":
                links[i] = 1

        return links


lf = PostHyphen()
lf.apply(reduced_ontonotes)


class NounPhrase(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        tokens = [t.text for t in instance['tokens']]

        verb_index = np.where(
            np.array([i for i in instance['verb_indicator']]) == 1)[0][0]
        verb = tokens[verb_index]
        for i in range(5):
            for c in range(len(tokens) - i):
                if tuple(tokens[c:c + i + 1]) in noun_phrases[verb]:
                    links[c + 1:c + i + 1] = [1] * (i)
        return links


lf = NounPhrase()
lf.apply(reduced_ontonotes)


class AndOr(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        tokens = [t.text for t in instance['tokens']]

        for i in range(len(instance['tokens']) - 1):
            if tokens[i] in {'and', 'or'}:
                links[i] = 1
                links[i + 1] = 1

        return links


lf = AndOr()
lf.apply(reduced_ontonotes)


class ConsecutiveCapitals(LinkingRule):
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

            if not all_caps and text[0].isupper(
            ) and instance['tokens'][i - 1].text[0].isupper():
                links[i] = 1

        return links


lf = ConsecutiveCapitals()
lf.apply(reduced_ontonotes)


class AdverbNoun(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        tokens = [t.text for t in instance['tokens']]
        token_dependencies = [t for t in instance['dependency'].metadata]

        for i, token_dependency in enumerate(token_dependencies):
            if token_dependency.pos_ == 'ADP' or token_dependency.dep_ == 'prep':
                if tokens[i] == token_dependency.text:
                    links[i + 1] = 1
                elif i + 1 < len(tokens) and tokens[i + 1] == token_dependency.text:
                    links[i + 1] = 1
        return links


lf = AdverbNoun()
lf.apply(reduced_ontonotes)


train_data = reduced_train
dev_data = reduced_dev
test_data = reduced_test


print(score_labels_majority_vote(test_data, span_level=True))
print('--------------------')

save_label_distribution('output/generative/dev_data.p', reduced_dev)
save_label_distribution('output/generative/test_data.p', reduced_test)

cnt = Counter()
for instance in ontonotes_docs:
    for tag in instance['tags']:
        cnt[tag] += 1

disc_label_to_ix = {value[0]: ix for ix, value in enumerate(cnt.most_common())}
gen_label_to_ix = {'ABS': 0, 'O': 1, 'I-ARG1': 2, 'I-ARG0': 3, 'I-ARGM-NEG': 4}


dist = get_mv_label_distribution(train_data, disc_label_to_ix, 'O')
save_label_distribution('output/generative/train_data_mv.p', train_data, dist)
dist = get_unweighted_label_distribution(train_data, disc_label_to_ix, 'O')
save_label_distribution(
    'output/generative/train_data_unweighted.p',
    train_data,
    dist)

epochs = 5


""" Naive Bayes Model"""
# Defines the model
tagging_rules, linking_rules = get_rules(train_data)
nb = NaiveBayes(
    len(gen_label_to_ix) - 1,
    len(tagging_rules),
    init_acc=0.9,
    acc_prior=50,
    balance_prior=10)

# Trains the model
config = LearningConfig()
config.batch_size = 16
p, r, f1 = train_generative_model(
    nb, train_data, dev_data, epochs, gen_label_to_ix, config)

# Evaluates the model
print('Naive Bayes: \n' + str(evaluate_generative_model(model=nb,
                                                        data=test_data, label_to_ix=gen_label_to_ix)))
print('--------------------')


# Saves the model
label_votes, link_votes, seq_starts = get_generative_model_inputs(train_data, gen_label_to_ix)

p_unary = nb.get_label_distribution(label_votes)
save_label_distribution(
    'output/generative/train_data_nb.p',
    train_data,
    p_unary,
    None,
    gen_label_to_ix,
    disc_label_to_ix)


""" HMM Model"""
# Defines the model
tagging_rules, linking_rules = get_rules(train_data)
hmm = HMM(
    len(gen_label_to_ix) - 1,
    len(tagging_rules),
    init_acc=0.9,
    acc_prior=100,
    balance_prior=500)

# Trains the model
config = LearningConfig()
config.batch_size = 16
p, r, f1 = train_generative_model(
    hmm, train_data, dev_data, epochs, gen_label_to_ix, config)

# Evaluates the model
print('HMM: \n' + str(evaluate_generative_model(model=hmm,
                                                data=test_data, label_to_ix=gen_label_to_ix)))
print('--------------------')


# Saves the model
label_votes, link_votes, seq_starts = get_generative_model_inputs(
    train_data, gen_label_to_ix)
p_unary, p_pairwise = hmm.get_label_distribution(label_votes, seq_starts)
save_label_distribution(
    'output/generative/train_data_hmm.p',
    train_data,
    p_unary,
    p_pairwise,
    gen_label_to_ix,
    disc_label_to_ix)


""" Linked HMM Model """
# Defines the model
tagging_rules, linking_rules = get_rules(train_data)
link_hmm = LinkedHMM(
    num_classes=len(gen_label_to_ix) - 1,
    num_labeling_funcs=len(tagging_rules),
    num_linking_funcs=len(linking_rules),
    init_acc=0.7,
    acc_prior=50,
    balance_prior=100)

# Trains the model
config = LearningConfig()
config.batch_size = 16
p, r, f1 = train_generative_model(
    link_hmm, train_data, dev_data, epochs=1, label_to_ix=gen_label_to_ix, config=config)

# Evaluates the model
print('Linked HMM: \n' + str(evaluate_generative_model(model=link_hmm,
                                                       data=test_data, label_to_ix=gen_label_to_ix)))
print('--------------------')


# Saves the model
inputs = get_generative_model_inputs(train_data, gen_label_to_ix)
p_unary, p_pairwise = link_hmm.get_label_distribution(*inputs)
save_label_distribution(
    'output/generative/train_data_link_hmm_1.p',
    train_data,
    p_unary,
    p_pairwise,
    gen_label_to_ix,
    disc_label_to_ix)
