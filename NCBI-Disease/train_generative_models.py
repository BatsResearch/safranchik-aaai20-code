from wiser.data.dataset_readers import NCBIDiseaseDatasetReader
from wiser.rules import TaggingRule, LinkingRule, UMLSMatcher, DictionaryMatcher
from wiser.generative import get_label_to_ix, get_rules
from labelmodels import *
from wiser.generative import train_generative_model
from labelmodels import LearningConfig
from wiser.generative import evaluate_generative_model
from wiser.data import save_label_distribution
from wiser.eval import *
from wiser.rules import ElmoLinkingRule
from collections import Counter

root = "../../data/"
reader = NCBIDiseaseDatasetReader()
train_data = reader.read('../data/NCBI/NCBItrainset_corpus.txt')
dev_data = reader.read('../data/NCBI/NCBIdevelopset_corpus.txt')
test_data = reader.read('../data/NCBI/NCBItestset_corpus.txt')


ncbi_docs = train_data + dev_data + test_data


dict_core = set()
dict_core_exact = set()
with open('../data/AutoNER_dicts/NCBI/dict_core.txt') as f:
    for line in f.readlines():
        line = line.strip().split()
        term = tuple(line[1:])

        if len(term) > 1 or len(term[0]) > 3:
            dict_core.add(term)
        else:
            dict_core_exact.add(term)

# Prepends common modifiers
to_add = set()
for term in dict_core:
    to_add.add(("inherited", ) + term)
    to_add.add(("Inherited", ) + term)
    to_add.add(("hereditary", ) + term)
    to_add.add(("Hereditary", ) + term)

dict_core |= to_add

print("✅ loaded dict_core")

# Removes common FP
dict_core_exact.remove(("WT1",))
dict_core_exact.remove(("VHL",))


dict_full = set()

with open('../data/AutoNER_dicts/NCBI/dict_full.txt') as f:
    for line in f.readlines():
        line = line.strip().split()
        dict_full.add(tuple(line))

print("✅ loaded dict_full")

lf = DictionaryMatcher(
    "CoreDictionaryUncased",
    dict_core,
    uncased=True,
    i_label="I")
lf.apply(ncbi_docs)

print("✅🏃‍♂️ run LF_DictionaryMatcher dict_core")


lf = DictionaryMatcher("CoreDictionaryExact", dict_core_exact, i_label="I")
lf.apply(ncbi_docs)

print("✅🏃‍♂️ run LF_DictionaryMatcher dict_core_exact")


class CancerLike(TaggingRule):
    def apply_instance(self, instance):
        tokens = [token.text.lower() for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        suffixes = ("edema", "toma", "coma", "noma")

        for i, token in enumerate(tokens):
            for suffix in suffixes:
                if token.endswith(suffix) or token.endswith(suffix + "s"):
                    labels[i] = 'I'
        return labels


lf = CancerLike()
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_CancerLike")


class CommonSuffixes(TaggingRule):

    suffixes = {
        "agia",
        "cardia",
        "trophy",
        "toxic",
        "itis",
        "emia",
        "pathy",
        "plasia"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            for suffix in self.suffixes:
                if instance['tokens'][i].lemma_.endswith(suffix):
                    labels[i] = 'I'
        return labels


lf = CommonSuffixes()
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_CommonSuffixes")


class Deficiency(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        # "___ deficiency"
        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                               1].lemma_ == 'deficiency':
                labels[i] = 'I'
                labels[i + 1] = 'I'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I'
                    else:
                        break

        # "deficiency of ___"
        for i in range(len(instance['tokens']) - 2):
            if instance['tokens'][i].lemma_ == 'deficiency' and instance['tokens'][i + 1].lemma_ == 'of':
                labels[i] = 'I'
                labels[i + 1] = 'I'
                nnp_active = False
                for j in range(i + 2, len(instance['tokens'])):
                    if instance['tokens'][j].pos_ in ('NOUN', 'PROPN'):
                        if not nnp_active:
                            nnp_active = True
                    elif nnp_active:
                        break
                    labels[j] = 'I'

        return labels


lf = Deficiency()
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_Deficiency")


class Disorder(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                               1].lemma_ == 'disorder':
                labels[i] = 'I'
                labels[i + 1] = 'I'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I'
                    else:
                        break

        return labels


lf = Disorder()
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_Disorder")


class Lesion(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                               1].lemma_ == 'lesion':
                labels[i] = 'I'
                labels[i + 1] = 'I'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I'
                    else:
                        break

        return labels


lf = Lesion()
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_Lesion")


class Syndrome(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                               1].lemma_ == 'syndrome':
                labels[i] = 'I'
                labels[i + 1] = 'I'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I'
                    else:
                        break

        return labels


lf = Syndrome()
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_Syndrome")


terms = []
with open('../data/umls/umls_body_part.txt', 'r') as f:
    for line in f.readlines():
        terms.append(line.strip().split(" "))
lf = DictionaryMatcher("TEMP", terms, i_label='TEMP', uncased=True, match_lemmas=True)
lf.apply(ncbi_docs)


class BodyTerms(TaggingRule):
    def apply_instance(self, instance):
        tokens = [token.text.lower() for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        terms = set([
            "cancer", "cancers",
            "damage",
            "disease", "diseases"
                       "pain",
            "injury", "injuries",
        ])

        for i in range(0, len(tokens) - 1):
            if instance['WISER_LABELS']['TEMP'][i] == 'TEMP':
                if tokens[i + 1] in terms:
                    labels[i] = "I"
                    labels[i + 1] = "I"
        return labels


lf = BodyTerms()
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_BodyTerms")


for doc in ncbi_docs:
    del doc['WISER_LABELS']['TEMP']


class OtherPOS(TaggingRule):
    other_pos = {"ADP", "ADV", "DET", "VERB"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(0, len(instance['tokens'])):
            if instance['tokens'][i].pos_ in self.other_pos:
                labels[i] = "O"
        return labels


lf = OtherPOS()
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_OtherPos")


stop_words = {"a", "as", "be", "but", "do", "even",
              "for", "from",
              "had", "has", "have", "i", "in", "is", "its", "just",
              "my", "no", "not", "on", "or",
              "that", "the", "these", "this", "those", "to", "very",
              "what", "which", "who", "with"}


class StopWords(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].lemma_ in stop_words:
                labels[i] = 'O'
        return labels


lf = StopWords()
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_StopWords")


class Punctuation(TaggingRule):

    other_punc = {".", ",", "?", "!", ";", ":", "(", ")",
                  "%", "<", ">", "=", "+", "/", "\\"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].text in self.other_punc:
                labels[i] = 'O'
        return labels


lf = Punctuation()
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_Punctuation")


class PossessivePhrase(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if instance['tokens'][i -
                                  1].text == "'s" or instance['tokens'][i].text == "'s":
                links[i] = 1

        return links


lf = PossessivePhrase()
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_PossessivePhrase")


class HyphenatedPhrase(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if instance['tokens'][i -
                                  1].text == "-" or instance['tokens'][i].text == "-":
                links[i] = 1

        return links


lf = HyphenatedPhrase()
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_HyphenatedPhrase")



lf = ElmoLinkingRule(.8)
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_ElmoLinkingRule")


class CommonBigram(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        tokens = [token.text.lower() for token in instance['tokens']]

        bigrams = {}
        for i in range(1, len(tokens)):
            bigram = tokens[i - 1], tokens[i]
            if bigram in bigrams:
                bigrams[bigram] += 1
            else:
                bigrams[bigram] = 1

        for i in range(1, len(tokens)):
            bigram = tokens[i - 1], tokens[i]
            count = bigrams[bigram]
            if count >= 6:
                links[i] = 1

        return links


lf = CommonBigram()
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_CommonBigram")


class ExtractedPhrase(LinkingRule):
    def __init__(self, terms):
        self.term_dict = {}

        for term in terms:
            term = [token.lower() for token in term]
            if term[0] not in self.term_dict:
                self.term_dict[term[0]] = []
            self.term_dict[term[0]].append(term)

        # Sorts the terms in decreasing order so that we match the longest
        # first
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
lf.apply(ncbi_docs)
print("✅🏃‍♂️ run LF_ExtractedPhrase")


print(score_labels_majority_vote(test_data, span_level=True))
print('🔥--------------------🔥')

save_label_distribution('output/generative/dev_data.p', dev_data)
save_label_distribution('output/generative/test_data.p', test_data)

cnt = Counter()
for instance in train_data + dev_data:
    for tag in instance['tags']:
        cnt[tag] += 1

disc_label_to_ix = {value[0]: ix for ix, value in enumerate(cnt.most_common())}
gen_label_to_ix = {'ABS': 0, 'I': 1, 'O': 2}

dist = get_mv_label_distribution(train_data, disc_label_to_ix, 'O')
save_label_distribution('output/generative/train_data_mv.p', train_data, dist)
dist = get_unweighted_label_distribution(train_data, disc_label_to_ix, 'O')
save_label_distribution(
    'output/generative/train_data_unweighted.p',
    train_data,
    dist)
print("✅🏃‍♂️ run save_label_distribution")

epochs = 5


""" Naive Bayes Model"""
# Defines the model
tagging_rules, linking_rules = get_rules(train_data)
nb = NaiveBayes(
    len(gen_label_to_ix) - 1,
    len(tagging_rules),
    init_acc=0.9,
    acc_prior=0.05,
    balance_prior=5.0)

# Trains the model
p, r, f1 = train_generative_model(
    nb, train_data, dev_data, epochs, gen_label_to_ix, LearningConfig())

# Evaluates the model
print('Naive Bayes: \n' + str(evaluate_generative_model(model=nb,
                                                        data=test_data, label_to_ix=gen_label_to_ix)))
print('--------------------')


# Saves the model
label_votes, link_votes, seq_starts = get_generative_model_inputs(
    train_data, gen_label_to_ix)
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
    acc_prior=50,
    balance_prior=500)

# Trains the model
p, r, f1 = train_generative_model(
    hmm, train_data, dev_data, epochs, label_to_ix=gen_label_to_ix, config=LearningConfig())

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
    init_acc=0.9,
    acc_prior=50,
    balance_prior=500)

# Trains the model
p, r, f1 = train_generative_model(
    link_hmm, train_data, dev_data, epochs, label_to_ix=gen_label_to_ix, config=LearningConfig())

# Evaluates the model
print('Linked HMM: \n' + str(evaluate_generative_model(model=link_hmm,
                                                       data=test_data, label_to_ix=gen_label_to_ix)))


# Saves the model
inputs = get_generative_model_inputs(train_data, gen_label_to_ix)
p_unary, p_pairwise = link_hmm.get_label_distribution(*inputs)
save_label_distribution(
    'output/generative/train_data_link_hmm.p',
    train_data,
    p_unary,
    p_pairwise,
    gen_label_to_ix,
    disc_label_to_ix)
