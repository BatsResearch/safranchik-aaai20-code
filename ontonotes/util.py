from allennlp.data.fields import ArrayField
from allennlp.data import Instance
from collections.abc import Iterable
import numpy as np
import pickle

# Returns objects for a given token, if any
def get_tree_modifiers(tokens, is_modifier=False, depth=0):

    if not isinstance(tokens, Iterable):
        tokens = [tokens]

    total = []
    for token in tokens:
        is_modifier_token = False

        if is_modifier_token or any([mod in token.dep_ for mod in {'mod', 'aux', 'mark'}]):
             is_modifier=True
        else:
            if depth > 0:
                continue

        leaves = [token] if depth > 0 else []
        left_leaves = get_tree_modifiers(token.lefts, is_modifier, depth+1)
        right_leaves = get_tree_modifiers(token.rights, is_modifier, depth+1)

        total += (left_leaves + leaves + right_leaves)
    return total

# Returns subject for a given token, if any
def get_tree_subject(tokens, is_subject=False, depth=0):

    if not isinstance(tokens, Iterable):
        tokens = [tokens]

    total = []
    for token in tokens:
        is_subject_token = False

        if (is_subject or 'subj' in token.dep_) and token.pos_ in {'NOUN', 'PRON', 'PROPN'}:
            is_subject_token = True
        else:
            if depth > 0:
                continue

        leaves = [token] if depth > 0 else []
        left_leaves = get_tree_subject(token.lefts, is_subject_token, depth+1)
        right_leaves = get_tree_subject(token.rights, is_subject_token, depth+1)

        total += (left_leaves + leaves + right_leaves)
    return total

# Returns objects for a given token, if any
def get_tree_object(tokens, is_object=False, depth=0):

    if not isinstance(tokens, Iterable):
        tokens = [tokens]

    total = []
    for token in tokens:
        is_object_token = False

        if (is_object or 'obj' in token.dep_) and token.pos_ in {'NOUN', 'PRON', 'PROPN'}:
            is_object_token = True
        else:
            if depth > 0:
                continue

        leaves = [token] if depth > 0 else []
        left_leaves = get_tree_object(token.lefts, is_object_token, depth+1)
        right_leaves = get_tree_object(token.rights, is_object_token, depth+1)

        total += (left_leaves + leaves + right_leaves)
    return total

# Returns negator objects for a given token, if any
def get_tree_negators(tokens, is_modifier=False, depth=0):

    if not isinstance(tokens, Iterable):
        tokens = [tokens]

    total = []
    for token in tokens:
        is_modifier_token = False

        if is_modifier_token or 'neg' in token.dep_:
             is_modifier=True
        else:
            if depth > 0:
                continue

        leaves = [token] if depth > 0 else []
        left_leaves = get_tree_negators(token.lefts, is_modifier, depth+1)
        right_leaves = get_tree_negators(token.rights, is_modifier, depth+1)

        total += (left_leaves + leaves + right_leaves)
    return total

# Returns objects for a given token, if any
def get_tree_modifiers(tokens, is_modifier=False, depth=0):

    if not isinstance(tokens, Iterable):
        tokens = [tokens]

    total = []
    for token in tokens:
        is_modifier_token = False

        if is_modifier_token or any([mod in token.dep_ for mod in {'mod', 'aux', 'mark'}]):
             is_modifier=True
        else:
            if depth > 0:
                continue

        leaves = [token] if depth > 0 else []
        left_leaves = get_tree_modifiers(token.lefts, is_modifier, depth+1)
        right_leaves = get_tree_modifiers(token.rights, is_modifier, depth+1)

        total += (left_leaves + leaves + right_leaves)
    return total


# Returns objects for a given token, if any
def get_subtree(tokens):


    if not isinstance(tokens, Iterable):
        tokens = [tokens]

    if tokens == []:
        return tokens

    total = []
    for token in tokens:
        if token.head.dep_ == 'ROOT' or token.head.head == 'ROOT' or 'obj' in token.head.dep_ or 'subj' in token.head.dep_:
            continue

        left_leaves = get_subtree(token.lefts)
        right_leaves = get_subtree(token.rights)
        total += left_leaves + [token] + right_leaves
    return total

# Returns first subject by recursively backtracking parents, if any
def get_subject(token, max_depth=0, depth=0):
    total = []

    if token.pos_ != 'VERB' or depth > max_depth:
        return total

    subject = get_tree_subject([token])

    if token.dep_ == 'ROOT':
        return subject

    if subject == []:
        subject = get_subject(token.head, max_depth, depth+1)

    return subject

# Returns first object by recursively backtracking parents, if any
def get_object(token, max_depth=0, depth=0):
    total = []

    if token.pos_ != 'VERB' or depth > max_depth:
        return total

    object = get_tree_object([token])

    if token.dep_ == 'ROOT':
        return object

    if object == []:
        object = get_object(token.head, max_depth, depth+1)

    return object
