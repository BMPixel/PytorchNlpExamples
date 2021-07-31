"""Pytorch Dataloader with CBOW method or Skip-gram.

Memory-friendly training pipeline using generator functions
Defines two dataset factory function get_cbow_dataset and get_skip_gram_dataset
Call get_dataloader("cbow") or get_dataloader("skip") to get a dataloader
"""

import re
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

vocab = None
tokenizer = get_tokenizer("basic_english")


class IterWrapper(IterableDataset):
    """A wrapper class that converts python generator into IterableDataset.
    
    :param generator: A generator instance
    """
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)


def initVocab(lines_limit=-1):
    """Intialize vocabulary from tokens in yield_tokens().

    :param lines_limit: Maximum lines used to construct the vocab, -1 means no limit, defaults to -1
    """
    global vocab, tokenizer
    print("Initializing vocabulary...")
    vocab = build_vocab_from_iterator(yield_tokens(lines_limit), specials=["<unk>"])
    vocab.set_default_index(vocab['<unk>'])
    print("Counted %d words in  dataset"%(len(vocab)))


def remove_html_tag(text):
    """Remove all html tag('<...>') and '\\n' in input text.

    :param text: Input string need processing
    :return: Processed string
    """
    dr = re.compile(r'<[^>]+>',re.S)
    dn = re.compile(r'\\n', re.S)
    return dn.sub('', dr.sub('',text))


def yield_tokens(lines_limit=-1):
    """Generator function that yield dataset

    Load yahoo dataset and yield all sentences from the train set
    in the form of token lists. limit to number of lines can be given to shrink
    the dataset down due to the full-scale dataset being too large.

    :param lines_limit: Maximum number of lines that can be yield, -1 means no limit, defaults to -1
    :yield: tokens of one sentence in the datatset in the form of string array like
            ["hello", "world", "!"]
    """
    print("Loading Yahoo dataset...")
    dataset = torchtext.datasets.YahooAnswers(split='train')
    accumulate_idx = 0
    accumulate = 0
    if not lines_limit == -1:
        # Allow accumulate_idx to increase to reach the number limit
        accumulate = 1 

    for _, text in dataset:
        yield tokenizer(remove_html_tag(text))

        accumulate_idx += accumulate
        if accumulate_idx >= lines_limit:
            break


def get_cbow_dataiter(window_size = 5, lines_limit=10000):
    """Generator function that yield all cbow training pairs.

    For each target word in the corpus, (window_size - 1) words around it are collected as
    context words, the result returns in the form of tuple.

    :param window_size: The length of scaning window where context words are acquired
                        the value is better to be odd as the target word lies in the middle
                        of the window, defaults to 5
    :param lines_limit: Maximum number of lines that can be yield, defaults to 10000
    :yield: a (context, target) pair generated with cbow method like 
            (tensor[576, 3435, 3434, 13], tensor[5])
    """
    left_b = window_size // 2
    right_b = window_size - left_b

    for tokens in yield_tokens(lines_limit):

        if len(tokens) < window_size:
            continue

        if not vocab:
            initVocab(lines_limit)
        indexes = vocab(tokens)

        for i in range(left_b, len(indexes) - right_b):
            target = torch.tensor(indexes[i], dtype=torch.int64)
            context = torch.tensor(indexes[i-left_b:i] + indexes[i:i+right_b], dtype=torch.int64)
            yield (context, target)


def get_skip_dataiter(window_size=5, lines_limit=10000):
    """Generator function that yield all skip_gram training pairs.

    For each original word in the corpus, (window_size - 1) words around it are collected as
    target words and finally become (window_size - 1) training pairs the result returns 
    in the form of tuple.

    :param window_size: The length of scaning window where target words are acquired
                        the value is better to be odd as the original word lies in the middle
                        of the window, defaults to 5
    :param lines_limit: Maximum number of lines that can be yield, defaults to 10000
    :yield: a (context, target) pair generated with skip_gram method like 
            (tensor[576], tensor[5])
    """
    left_b = window_size // 2
    right_b = window_size - left_b

    for tokens in yield_tokens(lines_limit):

        if len(tokens) < window_size: # Skips abnormally short sentence
            continue

        if not vocab:
            initVocab(lines_limit)
        indexes = vocab(tokens)

        for i in range(left_b, len(indexes) - right_b):
            origin = torch.tensor(indexes[i], dtype=torch.int64)
            for context in indexes[i-left_b:i] + indexes[i:i+right_b]:
                yield (origin, torch.tensor(context, dtype=torch.int64))


def get_dataloader(type="cbow", batch_size = 128, window_size=5, lines_limit=10000):
    """Returns a python dataloader with cbow method or skip-gram method.

    The return shape of cbow dataloader could be (tensor(batch_size * (window_size-1)), tensor(batch_size)) 
    with each row stands for
    a cbow training pair.
    The return shape of skip-gram dataloader could be (tensor(batch_size), tensor(batch_size)) 
    with each row stands for
    a skip-gram training pair.
    For a smaller dataset, cbow is recommended while skip-gram performs better in large corpus.
    Limit to number of lines can be given to shrinkthe dataset down 
    due to the full-scale dataset being too large.

    :param type: Which method to generate the training pair 'cbow' or 'skip', defaults to "cbow"
    :param batch_size: The size of batch in the dataloader, defaults to 128
    :param window_size: The length of scaning window where context words are acquired
                        the value is better to be odd as the original word lies in the middle
                        of the window, defaults to 5
    :param lines_limit: Maximum number of lines that can be yield, defaults to 10000
    :return: Dataloader instance
    """
    if not vocab:
            initVocab(lines_limit)

    if type == "cbow":
        train_iter = IterWrapper(get_cbow_dataiter(window_size, lines_limit))
        return DataLoader(
            train_iter,
            batch_size=batch_size,
            )
    elif type == "skip":
        train_iter = IterWrapper(get_skip_dataiter(window_size, lines_limit))
        return DataLoader(
            train_iter,
            batch_size=batch_size,
            )
    else:
        raise Exception("'type' should be either 'cbow' or 'skip' but got " + str(type))
            
