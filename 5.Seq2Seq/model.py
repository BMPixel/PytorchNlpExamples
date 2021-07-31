"""Sequence-to-sequence models.

Defines several Seq2seq models, there are three RNN modes(RNN, LSTM, GRU), 
two attention mode(on and off) and two bi_directional mode(on and off) for 
you to configurate and combine them freely.
Defined class Seq2seq, PackRNN, Seq2seqAtt
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Seq2seq(nn.Module):
    """Sequence-to-sequence without attention"""
    def __init__(self, vocab_size, hidden_size,
                core='RNN', bidirectional=False):
        """Sequence-to-sequence without attention.

        Provides parameters "core" and "bi-directional" to configurate what rnn
        (RNN, GRU, LSTM) to use and whether to make the rnn bi-directional.
        Because the task is character based, one_hot encoding is used instead of 
        word embedding.

        :param vocab_size: The number of words in vocabulary
        :param hidden_size: Size of hidden units
        :param core: Which rnn to use, RNN, LSTM or GRU, applied both to encoder and 
        decoder, defaults to "RNN"
        :param bidirectional: Use bi-directional rnn for encoder, defaults to False
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.core = core
        self.bidirectional = bidirectional
        bdf = 2 if bidirectional else 1 # bidirectional factor

        # Set up encoder/decoder on given core
        # decoder hidden_size must be timed with 2 to recieving bidirectional 
        # hidden units from encoder
        if core == 'RNN':
            self.encoder = nn.RNN(vocab_size, hidden_size, bidirectional=bidirectional)
            self.decoder = nn.RNN(vocab_size, hidden_size * bdf)
        elif core == 'GRU':
            self.encoder = nn.GRU(vocab_size, hidden_size, bidirectional=bidirectional)
            self.decoder = nn.GRU(vocab_size, hidden_size * bdf)
        else: # LSTM
            self.encoder = nn.LSTM(vocab_size, hidden_size, bidirectional=bidirectional)
            self.decoder = nn.LSTM(vocab_size, hidden_size * bdf)

        # Linear layer of every time step to generate outputs 
        self.out = nn.Linear(hidden_size * bdf, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, output):
        """Forward propagation with teacher forcing learning,

        Teacher forcing learning assumes the t-1 words when generating 
        t th word are all correct. So the output here doesn't means target.
        Output sequence needs to start with <SOS> token and target will 
        be t+1 th word in output given t th output word.

        :param input: Input tensors with shape [input_seq_length, batch_size]
        :param output: Output tensor with shape [output_seq_length, batch_size]
        :return: Possibilities of characters for every output with 
        shape[output_seq_length, batch_size, vocab_size]
        """
        batch_size = input.size(1)
        input = F.one_hot(input, num_classes=self.vocab_size).float()
        _, hidden = self.encoder(input) # hidden shape: [bdf, batch_size, hidden_size]

        # Rearrange dims of hidden_units, only matters in bidirectional mode
        # Turns shape of hidden from (2, batch_size, hidden_size)
        # to (batch_size, hidden_size * 2) and simply reshape is not ok
        # (2, batch_size, hidden_size) -> (batch_size, hidden_size, 2) -> 
        # (batch_size, hidden_size * 2) -> (1, batch_size, hidden_size * 2) is ok
        if self.core == 'RNN' or self.core == 'GRU':
            hidden = hidden.permute((1, 2, 0)).reshape((batch_size, -1)).unsqueeze(dim=0)
        else: # lstm return (h_0, c_0) as hidden
            h_0 = hidden[0].permute((1, 2, 0)).reshape((batch_size, -1)).unsqueeze(dim=0)
            c_0 = hidden[1].permute((1, 2, 0)).reshape((batch_size, -1)).unsqueeze(dim=0)
            hidden = (h_0, c_0)

        output = F.one_hot(output, num_classes=self.vocab_size).float()
        dec_hiddens, _ = self.decoder(output, hidden)
        preds = self.out(dec_hiddens)
        return self.softmax(preds)
    
    def predict(self, input, init_tensor, max_length=100):
        """Use self-regression to make a predict on given input sequence,
        only batch_size = 1 is allowed.

        :param input: Input tensor with shape [sentence_length, 1]
        :param init_tensor: First tensor to feeding into decoder with 
        shape[sentence_length=1, batch_size=1].
        :param max_length: Max prediction length it can return, defaults to 100
        """
        # Only (sentence_length, 1) is allowed to be fed in.(batch_size = 1)
        assert input.size(1) == 1

        input = F.one_hot(input, num_classes=self.vocab_size).float()
        _, hidden = self.encoder(input) # hidden shape: [1, batch_size, hidden_size]

        # Rearrange dims of hidden_units
        # Turns shape of hidden from (2, batch_size, hidden_size)
        # to (batch_size, hidden_size * 2) and simply reshape is not ok
        if self.core == 'RNN' or self.core == 'GRU':
            hidden = hidden.permute((1, 2, 0)).reshape((1, -1)).unsqueeze(dim=0)
        else: 
            # lstm return (h_0, c_0) as hidden
            h_0 = hidden[0].permute((1, 2, 0)).reshape((1, -1)).unsqueeze(dim=0)
            c_0 = hidden[1].permute((1, 2, 0)).reshape((1, -1)).unsqueeze(dim=0)
            hidden = (h_0, c_0)

        outputs = torch.zeros((max_length, 1, self.vocab_size))
        output = init_tensor # [length=1, batch_size=1]
        for i in range(max_length):
            output = F.one_hot(output, num_classes=self.vocab_size).float()
            _, hidden = self.decoder(output, hidden)
            output = self.out(hidden)
            outputs[i] = self.softmax(output)[0]
            output = output.argmax(dim=2)
        
        return outputs


class PackRNN(nn.Module):
    """RNN module with pack sequence to avoid using padding tokens"""
    def __init__(self, vocab_size, hidden_size):
        """RNN module with pack sequence to avoid using padding tokens.

        :param vocab_size: The size of vocabulary
        :param hidden_size: The numbers of hidden units used in rnn
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Use one-hot vector as input, no embedding layer
        self.encoder = nn.RNN(vocab_size, hidden_size)
        self.decoder = nn.RNN(vocab_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, output):
        """Forward propagation.

        :param input: vectorized input tensor list and lengths list
        :type input: (tensor(input_max_length, batch_size), list(int))
        :param output: vectorized input tensor list and lengths list
        :type output: (tensor(output_max_length, batch_size), list(int))
        """
        input, input_lengths = input
        output, output_lengths = output
        input = F.one_hot(input, num_classes=self.vocab_size).float()
        output = F.one_hot(output, num_classes=self.vocab_size).float()

        # Packs inputs and outputs into PackSequence
        # input/output_packed: (data=tensor(total_lengths, ), batch_sizes=list)
        # enforce_sorted is turned off because input sequences and output sequences
        # may have different sorting result. (Although in this dataset it won't go wrong)
        # Turning enforce_sorted off does harm to performance.
        input_packed = pack_padded_sequence(input, input_lengths, enforce_sorted=False)
        output_packed = pack_padded_sequence(output, output_lengths, enforce_sorted=False)

        # Enc_hidden shape:[1, batch_size, hidden_size]
        # dec_hiddens_packed is also a PackedSequence
        # dec_hiddens shape:[max_length, batch_size, hidden_size]
        _, enc_hidden = self.encoder(input_packed)
        dec_hiddens_packed, _ = self.decoder(output_packed, enc_hidden)
        dec_hiddens, _ = pad_packed_sequence(dec_hiddens_packed)

        preds = self.out(dec_hiddens)
        return self.softmax(preds)


class Seq2seqAtt(nn.Module):
    """Sequence-to-sequence with attention"""
    def __init__(self, vocab_size, hidden_size, 
                att_mode="general", core='RNN', bidirectional=False):
        """Sequence-to-sequence with attention

        Three attention modes is available, dot, general and concat:
        dot: makes inner product of enc_hidden and dec_hidden attention weight score
        general: makes enc_hidden * W * dec_hidden = score. Score is a scalar.
        concat: makes v'tanh(W * enc_hidden + U * dec_hidden) = score

        :param vocab_size: Size of vocabulary
        :param hidden_size: Size of hidden units in encoder and decoder
        :param att_mode: attention mode, 'general', 'dot' or 'concat', defaults to "general"
        :param core: rnn type of encoder and decoder, 'RNN', 'GRU' and 'LSTM', defaults to 'RNN'
        :param bidirectional: Use bidirectional mode in network, defaults to False
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.att_mode = att_mode
        self.core = core
        self.bidirectional = bidirectional
        bdf = 2 if bidirectional else 1 # bidirectional factor

        # Set up encoder/decoder on given core
        # decoder hidden_size must be timed with 2 to recieving bidirectional 
        # hidden units from encoder
        if core == 'RNN':
            self.encoder = nn.RNN(vocab_size, hidden_size, bidirectional=bidirectional)
            self.decoder = nn.RNN(vocab_size, hidden_size * bdf)
        elif core == 'GRU':
            self.encoder = nn.GRU(vocab_size, hidden_size, bidirectional=bidirectional)
            self.decoder = nn.GRU(vocab_size, hidden_size * bdf)
        else: # LSTM
            self.encoder = nn.LSTM(vocab_size, hidden_size, bidirectional=bidirectional)
            self.decoder = nn.LSTM(vocab_size, hidden_size * bdf)

        # The parameters used in attention calculation
        if att_mode == 'general':
            self.W = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        elif att_mode == 'concat':
            self.v = nn.Linear(1, self.hidden_size, bias=False)
            self.W = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.U = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Linear layer of every time step to generate outputs
        # forward encoder hidden, backward hidden(if needed) and attention values
        # will be fed into decoder.
        self.out = nn.Linear(hidden_size * bdf * 2, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, output, return_att=False):
        """Forward propagation function with atttention with teacher forcing.
        Correct feeding output sequence is required

        :param input: Input tensor with shape:[max_length, batch_size]
        :param output: Output tensor with shape:[max_length+1, batch_size]
        :param return_att: Return attention weights instead of prediction, only 
        used in testing, defaults to False
        :return: Possibilities of characters for every output with 
        shape[output_seq_length, batch_size, vocab_size]
        or the attention weights if return_att == True
        """
        batch_size = input.size(1)
        input = F.one_hot(input, num_classes=self.vocab_size).float()
        output = F.one_hot(output, num_classes=self.vocab_size).float()
        enc_hiddens, hidden = self.encoder(input) # hidden shape: [bdf, batch_size, hidden_size]

        # Rearrange dims of hidden_units, only matters in bidirectional mode
        # Turns shape of hidden from (2, batch_size, hidden_size)
        # to (batch_size, hidden_size * 2) and simply reshape is not ok
        # (2, batch_size, hidden_size) -> (batch_size, hidden_size, 2) -> 
        # (batch_size, hidden_size * 2) -> (1, batch_size, hidden_size * 2) is ok
        # with enc_hidden shape:[max_length, batch_size, 2 * hidden_size], no transform is ok
        if self.core == 'RNN' or self.core == 'GRU':
            hidden = hidden.permute((1, 2, 0)).reshape((batch_size, -1)).unsqueeze(dim=0)
        else: # lstm return (h_0, c_0) as hidden
            h_0 = hidden[0].permute((1, 2, 0)).reshape((batch_size, -1)).unsqueeze(dim=0)
            c_0 = hidden[1].permute((1, 2, 0)).reshape((batch_size, -1)).unsqueeze(dim=0)
            hidden = (h_0, c_0)

        # Compute attention value and output for every output time step.
        # The decoder here still works in vectorized fashion because computing
        # decoder hidden layers doesn't require attention value
        # Computing attention value with vetorized method
        # attention weight shape:[output_len, input_len, batch_size]
        # encoder hiddens shape:[input_len, batch_size, hidden_size]
        # attention shape:[output_len, batch_size, hidden_size]
        # So unsqueeze is needed in order to properly broadcast
        dec_hiddens, _ = self.decoder(output, hidden)
        att_weights = self.attention_weight(enc_hiddens, dec_hiddens, self.att_mode)
        att_weights = att_weights.unsqueeze(dim=3)
        enc_hiddens = enc_hiddens.unsqueeze(dim=0)
        attentions = torch.sum(enc_hiddens * att_weights, dim=1)
        preds = self.out(torch.cat([dec_hiddens, attentions], dim=2))

        # Returns attention weight instead of prediction
        # Only serve for the visualization when return_att == True
        if return_att:
            return att_weights.squeeze()
        else:
            return self.softmax(preds)

    def attention_weight(self, enc_hiddens, dec_hiddens, mode):
        """Gets global align weights respect to given encoding hiddens and decoding hidden

        All encoding hiddens should be passed in while only one decoder hidden is 
        needed in order to compute attention weights. However here all decoder hiddens
        are stil required because all attention values for all decoder hiddens will be 
        computed at once in order to utilize vectorization speed advantage. Simple and fast.

        :param enc_hiddens: Encoder hiddens tensor with shape:[input_len, batch_size, hidden_size]
        :param dec_hidden: Decoder hidden tensor with shape:[output_len, batch_size, hidden_size]
        :param mode: must be either 'inner', 'product' or 'addict'
        :return: attention weights with shape:[input_len, batch_size]
        """
        if mode == 'dot':
            # Vectorized scaled dot product, dec_hidden will be boardcast 
            # to shape [output_len, input_len, batch_size, hidden_size] and then
            # it will be summed along dim of hidden units(4th dim)
            # first decoder hiddens must be unsqueezed to (output_len, 1, batch_size, hidden_size)
            # for the sake of broadcasting.
            dec_hiddens = dec_hiddens.unsqueeze(dim=1)
            scores = torch.sum(enc_hiddens * dec_hiddens, dim=3) / math.sqrt(enc_hiddens.size(2))
        
        elif mode == 'general':
            # score = enc_hidden^T @ W @ dec_hidden
            energy = self.W(enc_hiddens)
            dec_hiddens = dec_hiddens.unsqueeze(dim=1)
            scores = torch.sum(energy * dec_hiddens, dim=3)

        elif mode == 'concat':
            # score = v^T tanh(W @ enc_hidden + U @ dec_hidden)
            # energy shape:[input_len, batch_size, hidden_size]
            # first make energy in shape of [output_len, input_len, batch_size, hidden_size]
            # then transform it to [output_len, input_len, batch_size, 1] and squeeze.
            dec_hiddens = dec_hiddens.unsqueeze(dim=1)
            energy = torch.tanh(self.W(enc_hiddens) + self.U(dec_hiddens))
            scores = self.v(energy).squeeze()
            
        else:
            raise ValueError(f"Mode {mode} is not defined!")

        # SoftMax along dim of input_len
        return F.softmax(scores, dim=1)


        
        