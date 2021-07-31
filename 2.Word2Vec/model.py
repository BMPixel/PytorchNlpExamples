"""Model classes of Word2Vec.

Provides several implements of Word2Vec module including SoftMax,
Hierarchical SoftMax(with complete binary tree)
and Negative sampling method
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2VecSM(nn.Module):
    """A Word2vec model using global softmax, works slowly.

    :param vocab_size: The number of words in the vocabulary
    :param embedding_dim: The dim of word embedding vector, defaults to 300
    """
    def __init__(self, vocab_size, embedding_dim=300):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Uses nn.EmbeddingBag in input because cbow inputs require average 
        # the embedded vectors of all context words, and embeddingbag does that
        self.input_embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim) # This is not the embedding matrix
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input):
        """Forward function.

        :param input: word indexes tensor with shape:[batch_size, 1]
        """
        embedded_in = self.input_embedding(input)

        # Make a inner production while keep batch organized.
        # Compute the score for every words in the vocab.
        # embedded_in.shape: (batch_size, embedding_dim)
        # target_embedding.shape : (vocab_size, embedding_dim)
        # Very computing expensive!
        score = embedded_in @ self.target_embedding.weight.T
        return self.softmax(score)


class Word2VecHSM(nn.Module):
    """A Word2vec model using hierarchical softmax, works faster but not that fast.

    Uses complete binary tree (not balanced tree or huffman tree); the model could be slow
    because of unhealthy tree struction.
    In practice, due to the depth of tree there may be accuracy underflow. So the loss computation
    (-log(P)) is intergrated into the model to use logarithm preventing accuracy underflow.

    :param vocab_size: The number of words in the vocabulary
    :param embedding_dim: The dim of word embedding vector, defaults to 300
    """
    def __init__(self, vocab_size, embedding_dim=300):

        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tree_scale = 2 ** math.ceil(math.log(self.vocab_size, 2))

        #Defines embedding matrix of original words
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim) 

        # Builds node tree
        # Each node is an logistic regression, with k-1 nodes for k leaves, vocab_size=k here.
        # Each leave is a integer standing for a token
        self.nodes = self._init_tree()
            
    def forward(self, input, target):
        """Forward function.

        :param input: input word indexes tensor with shape:[batch_size, 1]
        :param target: words whose conditional possibility need to be computed, same shape of inputs
        """
        batch_size = input.size(0)
        loss = torch.ones((batch_size, 1))
        embedded = self.embedding(input)

        # Products up all node's output in the path all the way to target token
        for i in range(batch_size):
            for node_idx, go_right in self._get_path(target[i].item()):
                # Skips "None" node
                if not self.nodes[node_idx]:
                    continue

                if go_right: 
                    # print(self.nodes[node_idx](embedded[i]))
                    loss[i] += -torch.log(self.nodes[node_idx](embedded[i]))[0]
                else:
                    # print(embedded.shape)
                    # print(self.nodes[node_idx](embedded[i]).shape)
                    loss[i] += -torch.log(1 - self.nodes[node_idx](embedded[i]))[0]
            
        return loss.mean()
    
    def _init_tree(self):
        node_list = nn.ModuleList([None] * (self.tree_scale + 1))

        # Basic idea: For any node with two children, use a logisitic regression;
        # For any node with only one children, use f(x) = 0 (Mean this node has zero 
        # possibility to go to right children) as the node.
        # According to the nature of complete binary tree, any node with one children
        # could only has one left children and no right children, and all of these node
        # appear in the path between root and rightest leave.
        bound_path = self._get_path(self.vocab_size-1)
        for i, d in bound_path:
            if d == 1:
                node_list[i] = nn.Sequential(
                    nn.Linear(self.embedding_dim, 1, bias=False),
                    nn.Sigmoid()
                    )
            # 2**int(math.log(i, 2)) indicate the left bound index of nodes whose depth
            # equal to i. Fills with small network between 2**int(math.log(i, 2)) and i
            for j in range(2**int(math.log(i, 2)), i):
                node_list[j] = nn.Sequential(
                    nn.Linear(self.embedding_dim, 1, bias=False),
                    nn.Sigmoid()
                    )
        return node_list

    def _get_path(self, end):
        """Computes a way from root to end"""
        path_nodes = []
        end += self.tree_scale
        while end > 1:
            parent = end // 2
            direction = end - parent*2 # 1 for right, 0 for left
            end //= 2
            path_nodes.insert(0, (parent, direction))
        
        return path_nodes


class Word2VecNSLoss(nn.Module):
    """A class compute loss of word2vec model with negative smapling

    Input_indexes, positive_indexes and negative_indexes need to be feed into
    the model at the same time for it can average all loss on every positive
    and negative words.

    :param vocab_size: The number of words in the vocabulary
    :param embedding_dim: The dim of word embedding vector, defaults to 300
    """
    def __init__(self, vocab_size, embedding_dim=300):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.input_embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.context_embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
    
    def forward(self, input, pos_con, neg_con):
        """Forward function computes loss

        :param input: Input word indexes tensor with shape:[batch, 1]
        :param pos_con: Context word indexes tensor with shape:[batch, 1]
        :param neg_con: Irrelevant word indexes tensor with shape:[batch * K, 1]
        :return: loss
        """
        K = neg_con.size(0) // input.size(0)

        embedded = self.input_embedding(input)
        embedded_exp = torch.cat([embedded] * K, dim=0)
        pos_embedded = self.context_embedding(pos_con)
        neg_embedded = self.context_embedding(neg_con)
        pos_score = (pos_embedded * embedded).sum(dim=1, keepdim=True)
        neg_score = (neg_embedded * embedded_exp).sum(dim=1, keepdim=True)
        combine = torch.cat([pos_score, -neg_score], dim = 0)

        return -torch.log(torch.sigmoid(combine)).mean()

        





