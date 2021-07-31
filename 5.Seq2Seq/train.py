"""Training functions.

Training function and evaluation function
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from tqdm import tqdm
from utils import Timer


def train(model, train_loader, num_epoches, log_interval=60, packed=False):
    """Trains for a given recurrent model

    :param model: The model need to be trained, must be either Seq2Seq, PackRNN, Seq2SeqAtt
    from model.py.
    :param train_loader: Iterable, reusable(no generator) train_loader that contains of batches.
    :param num_epoches: The number of epoches it will iterate.
    :param log_interval: The interval iters of logging information, defaults to 60
    :param packed: True if PackRNN as model, means lengths parameter is needed when 
    forward propagation, defaults to False
    """
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    Timer.Start(len(train_loader) * num_epoches)
    for epoch in range(num_epoches):
        running_loss = 0
        for idx, (input, output, target) in enumerate(train_loader):
            # Calculate lengths list of input and output
            input_lengths = torch.tensor(list(map(len, input)))
            output_lengths = torch.tensor(list(map(len, output)))

            # Shape list(tensors) into tensor with shape[max_length, batch_size]
            input = pad_sequence(input)
            output = pad_sequence(output)
            target = pad_sequence(target)

            # Output sequence doesn't have impacts on final accuracy.
            # It's ok to use it when compute accuracy
            if packed:
                preds = model((input, input_lengths), (output, output_lengths))
            else:
                preds = model(input, output)

            # input shape: [sentence_length, batch_size]
            # prediction shape: [sentence_length, batch_size, vocab_size], must be [x, vocab_size] 
            # to feed in loss function
            max_length, batch_size, vocab_size = preds.shape
            loss = criterion(preds.view((-1, vocab_size)), target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Regularly prints training information
            running_loss += loss.item()
            if idx % log_interval == log_interval - 1:
                print("Epoch: %d\tStep: %d\tLoss=%.3f\t%s" % (
                    epoch + 1,
                    idx, 
                    running_loss / log_interval,
                    Timer.Remain(len(train_loader) * epoch + idx)
                    ))
                running_loss = 0
    Timer.Stop()


def evaluate(model, dev_loader, packed=False):
    """Evaluate given model on given dataloader"""
    correct_total = 0
    lengths_total = 0

    for input, output, target in tqdm(dev_loader):
        input_lengths = list(map(len, input))
        output_lengths = list(map(len, output))
        input = pad_sequence(input)
        output = pad_sequence(output)
        target = pad_sequence(target)

        # Input shape:[max_length, batch_size]
        # Output, Target shape:[max_length+1, batch_size]
        # Prediction shape:[max_length+1, batch_size, vocab_size]
        if packed:
            preds = model((input, input_lengths), (output, output_lengths))
        else:
            preds = model(input, output)
        preds = preds.argmax(dim=2)
        _, batch_size = input.shape

        for i in range(batch_size):
            lengths_total += output_lengths[i]
            correct_total += torch.sum(preds[:output_lengths[i], i] == target[:output_lengths[i], i])
            # print(tensor2str(preds[:lengths[i], i].view(-1)),
            #     tensor2str(target[:lengths[i], i].view(-1)),
            #     torch.sum(preds[:lengths[i], i] == target[:lengths[i], i]),
            #     lengths[i])        
    return correct_total / lengths_total