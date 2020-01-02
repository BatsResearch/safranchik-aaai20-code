from wiser.data.dataset_readers.cdr import CDRCombinedDatasetReader
from wiser.lf import LabelingFunction, LinkingFunction, UMLSMatcher, DictionaryMatcher
from wiser.generative import get_label_to_ix, get_rules
from labelmodels import *
from wiser.generative import train_generative_model
from labelmodels import LearningConfig
from wiser.generative import evaluate_generative_model
from wiser.data import save_label_distribution
from wiser.eval import *
import pickle


# Loads Data
cdr_reader = CDRCombinedDatasetReader()
train_data = cdr_reader.read('../data/BC5CDR/CDR_TrainingSet.BioC.xml')
dev_data = cdr_reader.read('../data/BC5CDR/CDR_DevelopmentSet.BioC.xml')
test_data = cdr_reader.read('../data/BC5CDR/CDR_TestSet.BioC.xml')
cdr_docs = train_data + dev_data + test_data

# Applies Dictionary Matcher Functions
dict_core_chem = set()
dict_core_chem_exact = set()
dict_core_dis = set()
dict_core_dis_exact = set()

with open('../data/autoner_dicts/BC5CDR/dict_core.txt') as f:
    for line in f.readlines():
        line = line.strip().split(None, 1)
        entity_type = line[0]
        tokens = cdr_reader.get_tokenizer()(line[1])
        term = tuple([str(x) for x in tokens])

        if len(term) > 1 or len(term[0]) > 3:
            if entity_type == 'Chemical':
                dict_core_chem.add(term)
            elif entity_type == 'Disease':
                dict_core_dis.add(term)
            else:
                raise Exception()
        else:
            if entity_type == 'Chemical':
                dict_core_chem_exact.add(term)
            elif entity_type == 'Disease':
                dict_core_dis_exact.add(term)
            else:
                raise Exception()

lf = DictionaryMatcher("DictCore-Chemical", dict_core_chem,
                       i_label="I-Chemical", uncased=True)
lf.apply(cdr_docs)
lf = DictionaryMatcher("DictCore-Chemical-Exact",
                       dict_core_chem_exact, i_label="I-Chemical", uncased=False)
lf.apply(cdr_docs)
lf = DictionaryMatcher("DictCore-Disease", dict_core_dis,
                       i_label="I-Disease", uncased=True)
lf.apply(cdr_docs)
lf = DictionaryMatcher("DictCore-Disease-Exact",
                       dict_core_dis_exact, i_label="I-Disease", uncased=False)
lf.apply(cdr_docs)


terms = []
with open('../data/umls/umls_element_ion_or_isotope.txt', 'r') as f:
    for line in f.readlines():
        terms.append(line.strip().split(" "))
lf = DictionaryMatcher("Element, Ion, or Isotope", terms,
                       i_label='I-Chemical', uncased=True, match_lemmas=True)
lf.apply(cdr_docs)


terms = []
with open('../data/umls/umls_organic_chemical.txt', 'r') as f:
    for line in f.readlines():
        terms.append(line.strip().split(" "))
lf = DictionaryMatcher("Organic Chemical", terms,
                       i_label='I-Chemical', uncased=True, match_lemmas=True)
lf.apply(cdr_docs)


terms = []
with open('../data/umls/umls_antibiotic.txt', 'r') as f:
    for line in f.readlines():
        terms.append(line.strip().split(" "))
lf = DictionaryMatcher("Antibiotic", terms,
                       i_label='I-Chemical', uncased=True, match_lemmas=True)
lf.apply(cdr_docs)


terms = []
with open('../data/umls/umls_disease_or_syndrome.txt', 'r') as f:
    for line in f.readlines():
        terms.append(line.strip().split(" "))
lf = DictionaryMatcher("Disease or Syndrome", terms,
                       i_label='I-Disease', uncased=True, match_lemmas=True)
lf.apply(cdr_docs)


# ## Applies Other Labeling Functions
terms = []
with open('../data/umls/umls_body_part.txt', 'r') as f:
    for line in f.readlines():
        terms.append(line.strip().split(" "))
lf = DictionaryMatcher("TEMP", terms, i_label='TEMP',
                       uncased=True, match_lemmas=True)
lf.apply(cdr_docs)


class BodyTerms(LabelingFunction):
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

        for i in range(0, len(tokens)-1):
            if instance['WISER_LABELS']['TEMP'][i] == 'TEMP':
                if tokens[i+1] in terms:
                    labels[i] = "I-Disease"
                    labels[i+1] = "I-Disease"
        return labels


lf = BodyTerms()
lf.apply(cdr_docs)


for doc in cdr_docs:
    del doc['WISER_LABELS']['TEMP']


class Acronyms(LabelingFunction):
    other_lfs = {
        'I-Chemical': ("Antibiotic", "Element, Ion, or Isotope", "Organic Chemical"),
        'I-Disease':  ("BodyTerms", "Disease or Syndrome")
    }

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        active = False
        for tag, lf_names in self.other_lfs.items():
            acronyms = set()
            for lf_name in lf_names:
                for i in range(len(instance['tokens']) - 2):
                    if instance['WISER_LABELS'][lf_name][i] == tag:
                        active = True
                    elif active and instance['tokens'][i].text == '(' and instance['tokens'][i+2].pos_ == "PUNCT" and instance['tokens'][i+1].pos_ != "NUM":
                        acronyms.add(instance['tokens'][i+1].text)
                        active = False
                    else:
                        active = False

            for i, token in enumerate(instance['tokens']):
                if token.text in acronyms:
                    labels[i] = tag

        return labels


lf = Acronyms()
lf.apply(cdr_docs)


class Damage(LabelingFunction):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])-1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i+1].lemma_ == 'damage':
                labels[i] = 'I-Disease'
                labels[i+1] = 'I-Disease'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I-Disease'
                    else:
                        break

        return labels


lf = Damage()
lf.apply(cdr_docs)


# In[12]:


class Disease(LabelingFunction):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])-1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i+1].lemma_ == 'disease':
                labels[i] = 'I-Disease'
                labels[i+1] = 'I-Disease'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I-Disease'
                    else:
                        break

        return labels


lf = Disease()
lf.apply(cdr_docs)


class Disorder(LabelingFunction):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i+1].lemma_ == 'disorder':
                labels[i] = 'I-Disease'
                labels[i+1] = 'I-Disease'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I-Disease'
                    else:
                        break

        return labels


lf = Disorder()
lf.apply(cdr_docs)


class Lesion(LabelingFunction):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])-1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i+1].lemma_ == 'lesion':
                labels[i] = 'I-Disease'
                labels[i+1] = 'I-Disease'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I-Disease'
                    else:
                        break

        return labels


lf = Lesion()
lf.apply(cdr_docs)


class Syndrome(LabelingFunction):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])-1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i+1].lemma_ == 'syndrome':
                labels[i] = 'I-Disease'
                labels[i+1] = 'I-Disease'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I-Disease'
                    else:
                        break

        return labels


lf = Syndrome()
lf.apply(cdr_docs)


exceptions = {'determine', 'baseline', 'decline',
              'examine', 'pontine', 'vaccine',
              'routine', 'crystalline', 'migraine',
              'alkaline', 'midline', 'borderline',
              'cocaine', 'medicine', 'medline',
              'asystole', 'control', 'protocol',
              'alcohol', 'aerosol', 'peptide',
              'provide', 'outside', 'intestine',
              'combine', 'delirium', 'VIP'}

suffixes = ('ine', 'ole', 'ol', 'ide', 'ine', 'ium', 'epam')


class ChemicalSuffixes(LabelingFunction):
    def apply_instance(self, instance):

        labels = ['ABS'] * len(instance['tokens'])

        acronyms = set()
        for i, t in enumerate(instance['tokens']):
            if len(t.lemma_) >= 7 and t.lemma_ not in exceptions and t.lemma_.endswith(suffixes):
                labels[i] = 'I-Chemical'

                if i < len(instance['tokens'])-3 and instance['tokens'][i+1].text == '(' and instance['tokens'][i+3].text == ')':
                    acronyms.add(instance['tokens'][i+2].text)

        for i, t in enumerate(instance['tokens']):
            if t.text in acronyms and t.text not in exceptions:
                labels[i] = 'I-Chemical'
        return labels


lf = ChemicalSuffixes()
lf.apply(cdr_docs)


class CancerLike(LabelingFunction):
    def apply_instance(self, instance):
        tokens = [token.text.lower() for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        suffixes = ("edema", "toma", "coma", "noma")

        for i, token in enumerate(tokens):
            for suffix in suffixes:
                if token.endswith(suffix) or token.endswith(suffix + "s"):
                    labels[i] = 'I-Disease'
        return labels


lf = CancerLike()
lf.apply(cdr_docs)


exceptions = {'diagnosis', 'apoptosis', 'prognosis', 'metabolism'}

suffixes = ("agia", "cardia", "trophy", "itis",
            "emia", "enia", "pathy", "plasia", "lism", "osis")


class DiseaseSuffixes(LabelingFunction):
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i, t in enumerate(instance['tokens']):
            if len(t.lemma_) >= 5 and t.lemma_.lower() not in exceptions and t.lemma_.endswith(suffixes):
                labels[i] = 'I-Disease'

        return labels


lf = DiseaseSuffixes()
lf.apply(cdr_docs)


exceptions = {'hypothesis', 'hypothesize', 'hypobaric', 'hyperbaric'}

prefixes = ('hyper', 'hypo')


class DiseasePrefixes(LabelingFunction):
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i, t in enumerate(instance['tokens']):
            if len(t.lemma_) >= 5 and t.lemma_.lower() not in exceptions and t.lemma_.startswith(prefixes):
                if instance['tokens'][i].pos_ == "NOUN":
                    labels[i] = 'I-Disease'

        return labels


lf = DiseasePrefixes()
lf.apply(cdr_docs)


exceptions = {"drug", "pre", "therapy", "anesthetia",
              "anesthetic", "neuroleptic", "saline", "stimulus"}


class Induced(LabelingFunction):
    def apply_instance(self, instance):

        labels = ['ABS'] * len(instance['tokens'])

        for i in range(1, len(instance['tokens'])-3):
            lemma = instance['tokens'][i].lemma_.lower()
            if instance['tokens'][i].text == '-' and instance['tokens'][i+1].lemma_ == 'induce':
                labels[i] = 'O'
                labels[i+1] = 'O'
                if instance['tokens'][i-1].lemma_ in exceptions or instance['tokens'][i-1].pos_ == "PUNCT":
                    labels[i-1] = 'O'
                else:
                    labels[i-1] = 'I-Chemical'
        return labels


lf = Induced()
lf.apply(cdr_docs)


class Vitamin(LabelingFunction):
    def apply_instance(self, instance):

        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])-1):
            text = instance['tokens'][i].text.lower()
            if instance['tokens'][i].text.lower() == 'vitamin':
                labels[i] = 'I-Chemical'
                if len(instance['tokens'][i+1].text) <= 2 and instance['tokens'][i+1].text.isupper():
                    labels[i+1] = 'I-Chemical'

        return labels


lf = Vitamin()
lf.apply(cdr_docs)


class Acid(LabelingFunction):
    def apply_instance(self, instance):

        labels = ['ABS'] * len(instance['tokens'])

        tokens = instance['tokens']

        for i, t in enumerate(tokens):
            if i > 0 and t.text.lower() == 'acid' and tokens[i-1].text.endswith('ic'):
                labels[i] = 'I-Chemical'
                labels[i-1] = 'I-Chemical'

        return labels


lf = Acid()
lf.apply(cdr_docs)


class OtherPOS(LabelingFunction):
    other_pos = {"ADP", "ADV", "DET", "VERB"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(0, len(instance['tokens'])):
            # Some chemicals with long names get tagged as verbs
            if instance['tokens'][i].pos_ in self.other_pos and instance['WISER_LABELS']['Organic Chemical'][i] == 'ABS' and instance['WISER_LABELS']['DictCore-Chemical'][i] == 'ABS':
                labels[i] = "O"
        return labels


lf = OtherPOS()
lf.apply(cdr_docs)


stop_words = {"a", "an", "as", "be", "but", "do", "even",
              "for", "from",
              "had", "has", "have", "i", "in", "is", "its", "just",
              "may", "my", "no", "not", "on", "or",
              "than", "that", "the", "these", "this", "those", "to", "very",
              "what", "which", "who", "with"}


class StopWords(LabelingFunction):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].lemma_ in stop_words:
                labels[i] = 'O'
        return labels


lf = StopWords()
lf.apply(cdr_docs)


class CommonOther(LabelingFunction):
    other_lemmas = {'patient', '-PRON-', 'induce', 'after', 'study',
                    'rat', 'mg', 'use', 'treatment', 'increase',
                    'day', 'group', 'dose', 'treat', 'case', 'result',
                    'kg', 'control', 'report', 'administration', 'follow',
                    'level', 'suggest', 'develop', 'week', 'compare',
                    'significantly', 'receive', 'mouse',
                    'protein', 'infusion', 'output', 'area', 'effect',
                    'rate', 'weight', 'size', 'time', 'year',
                    'clinical', 'conclusion', 'outcome', 'man', 'woman',
                    'model', 'concentration'}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].lemma_ in self.other_lemmas:
                labels[i] = 'O'
        return labels


lf = CommonOther()
lf.apply(cdr_docs)


class Punctuation(LabelingFunction):

    other_punc = {"?", "!", ";", ":", ".", ",",
                  "%", "<", ">", "=", "\\"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].text in self.other_punc:
                labels[i] = 'O'
        return labels


lf = Punctuation()
lf.apply(cdr_docs)


# ## Applies Linking Functions
class PossessivePhrase(LinkingFunction):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if instance['tokens'][i-1].text == "'s" or instance['tokens'][i].text == "'s":
                links[i] = 1

        return links


lf = PossessivePhrase()
lf.apply(cdr_docs)


class HyphenatedPrefix(LinkingFunction):
    chem_mods = set(["alpha", "beta", "gamma", "delta", "epsilon"])

    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if (instance['tokens'][i-1].text.lower() in self.chem_mods or
                    len(instance['tokens'][i-1].text) < 2) \
                    and instance['tokens'][i].text == "-":
                links[i] = 1

        return links


lf = HyphenatedPrefix()
lf.apply(cdr_docs)


class PostHyphen(LinkingFunction):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if instance['tokens'][i-1].text == "-":
                links[i] = 1

        return links


lf = PostHyphen()
lf.apply(cdr_docs)


dict_full = set()

with open('../data/autoner_dicts/BC5CDR/dict_full.txt') as f:
    for line in f.readlines():
        tokens = cdr_reader.get_tokenizer()(line.strip())
        term = tuple([str(x) for x in tokens])
        if len(term) > 1:
            dict_full.add(tuple(term))


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
lf.apply(cdr_docs)


# Trains, Evaluates, and Saves Generative Models to Disk
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
nb = NaiveBayes(len(gen_label_to_ix)-1, len(tagging_rules),
                init_acc=0.9, acc_prior=0.01, balance_prior=5.0)

# Trains the model
p, r, f1 = train_generative_model(
    nb, train_data, dev_data, epochs, gen_label_to_ix, LearningConfig())

# Evaluates the model
print('NB: \n' + str(evaluate_generative_model(model=nb,
                                               data=test_data, label_to_ix=gen_label_to_ix)))
print('--------------------')

# Saves the model
label_votes, link_votes, seq_starts = get_generative_model_inputs(
    train_data, gen_label_to_ix)
p_unary = nb.get_label_distribution(label_votes)
save_label_distribution('output/train_data_nb.p', train_data,
                        p_unary, None, gen_label_to_ix, disc_label_to_ix)


""" HMM Model"""
# Defines the model
gen_label_to_ix, disc_label_to_ix = get_label_to_ix(train_data)
tagging_rules, linking_rules = get_rules(train_data)
hmm = HMM(len(gen_label_to_ix)-1, len(tagging_rules),
          init_acc=0.9, acc_prior=5, balance_prior=500)

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
save_label_distribution('output/train_data_hmm.p', train_data,
                        p_unary, p_pairwise, gen_label_to_ix, disc_label_to_ix)


""" Linked HMM Model """
# Defines the model
gen_label_to_ix, disc_label_to_ix = get_label_to_ix(train_data)
tagging_rules, linking_rules = get_rules(train_data)
link_hmm = LinkedHMM(num_classes=len(gen_label_to_ix)-1, num_labeling_funcs=len(tagging_rules),
                     num_linking_funcs=len(linking_rules), init_acc=0.9, acc_prior=5, balance_prior=500)

# Trains the model
p, r, f1 = train_generative_model(
    link_hmm, train_data, dev_data, epochs, label_to_ix=gen_label_to_ix, config=LearningConfig())

# Evaluates the model
print('Linked HMM: \n' + str(evaluate_generative_model(model=link_hmm,
                                                       data=test_data, label_to_ix=gen_label_to_ix)))

# Saves the model
inputs = get_generative_model_inputs(train_data, gen_label_to_ix)
p_unary, p_pairwise = link_hmm.get_label_distribution(*inputs)
save_label_distribution('output/train_data_link_hmm.p', train_data,
                        p_unary, p_pairwise, gen_label_to_ix, disc_label_to_ix)
