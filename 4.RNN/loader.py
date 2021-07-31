"""Pytorch Dataloader of CoNLL2000Chunking dataset.

Not-memory-friendly dataloader functions.
Call get_dataloader(mode="train") to get a train dataloader and 
get_dataloader(mode="test") to get one for test
Call print_format(text, tags, pred) to visualize text
"""
import torch
import torchtext
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torch.utils.data import DataLoader

data_set = None
vocab_text = None
vocab_tag = None
length_input = 30


def init_dataset():
    """Initialize dataset of Part-of-speech segments tags dataset of WSJ"""
    global data_set
    print("Loading dataset...")
    data_iter = torchtext.datasets.CoNLL2000Chunking(split='train')
    data_set = list(data_iter)


def init_vocabs():
    """Initialize vocabulary, also initialize dataset if needed"""
    global vocab_text, vocab_tag, data_set
    if not data_set:
        init_dataset()

    counter_text = Counter()
    counter_tag = Counter()
    for line, b_tags, i_tags in data_set:
        counter_text.update(line)
        counter_tag.update(i_tags)
    counter_tag.update(['PAD'])
    counter_text.update(['<unk>', '<pad>'])
    vocab_text = vocab(OrderedDict(counter_text.most_common()))
    vocab_tag = vocab(OrderedDict(counter_tag.most_common()))
    vocab_text.set_default_index(vocab_text['<unk>'])

    # Prints out some information
    print("Counted %d words in dataset \n Counted %d output classes" % (len(vocab_text), len(vocab_tag)))


def collate_batch(batch):
    """Collates list of some items in dataset into a well-organized batch"""
    if not vocab_text:
        init_vocabs()
    batch_size = len(batch)
    targets = torch.zeros((batch_size, length_input))
    inputs = torch.zeros((batch_size, length_input))
    
    for i, item in enumerate(batch):
        text, _, tags = item
        length_sentence = len(text)
        for j in range(length_input):
            if j < length_sentence:
                targets[i, j] = vocab_tag[tags[j]]
                inputs[i, j] = vocab_text[text[j]]
            else:
                targets[i, j] = vocab_tag['PAD']
                inputs[i, j] = vocab_text['<pad>']
        
    # The return shape should be [sentence_length, batch_size]
    return inputs.T.contiguous().long(), targets.T.contiguous().long()


def get_dataloader(batch_size=128, sentence_length=30, mode="train", split_ratio=0.9):
    """Get a pytorch dataloader

    :param batch_size: Size of batch, defaults to 128
    :param sentence_length: length of all sentences, sentences 
    with different lengthes will be cut or padded, defaults to 30
    :param mode: dataset mode "train" or "test", defaults to "train"
    :param split_ratio: the split ratio of train_set and test_set, defaults to 0.9
    """
    global length_input
    length_input = sentence_length

    if not vocab_text:
        init_vocabs()
    
    train_size = int(len(data_set) * split_ratio)
    if mode == "train":
        return DataLoader(data_set[:train_size], batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    else:
        return DataLoader(data_set[train_size:], batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


def print_format(input_indexes, tags_indexes, pred_indexes=None):
    """Prints text string and corresponding tags from tensors, 
    pred_tensor from model is optional.

    >>> text = torch.tensor([115, 1914, 1341, 2], dtype=torch.int64)
    >>> tags = torch.tensor([1, 3, 1, 2], dtype=torch.int64)
    >>> pred = torch.tensor([1, 5, 1, 2], dtype=torch.int64)
    >>> print_format(text, tags, pred)
     He /credits/imports/.
    B-NP --B-VP- --B-NP- O
    B-NP --I-VP- --B-NP- O

    :param input_indexes: Indexes list of input text
    :type input_indexes: List or tensor(text_length, )
    :param tags_indexes: Indexes list of correct tags
    :type tags_indexes: List or tensor(text_length, )
    :param pred_indexes: Indexes list of tags prediction, defaults to None
    :type pred_indexes: List or tensor(text_length, ), optional
    """
    text_list = vocab_text.lookup_tokens(list(input_indexes))
    tags_list = vocab_tag.lookup_tokens(list(tags_indexes))
    if pred_indexes is not None:
        pred_list = vocab_tag.lookup_tokens(list(pred_indexes))

    # Input lists must have same length
    assert len(text_list) == len(tags_list)

    # PAD tokens shouldn't be printed, shorten text length to prevent that
    print_length = 0
    while print_length < len(text_list) and text_list[print_length] != "<pad>":
        print_length += 1

    # Aligns all tokens for text in batch
    for i in range(print_length):
        if pred_indexes is not None:
            max_len = max(len(text_list[i]), len(tags_list[i]), len(pred_list[i]))
        else:
            max_len = max(len(text_list[i]), len(tags_list[i]))
        text_list[i] = text_list[i].center(max_len)
        tags_list[i] = tags_list[i].center(max_len, '-')
        if pred_indexes is not None:
            pred_list[i] = pred_list[i].center(max_len, '-')

    print("/".join(text_list[:print_length]))
    print(" ".join(tags_list[:print_length]))
    if pred_indexes is not None:
        print(" ".join(pred_list[:print_length]))