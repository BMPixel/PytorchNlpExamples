"""Training functions.

Provides train(model, data_loader) and evaluate(model, data_loader) for training and testing.
Model could be either RNN, LSTM, Bi-LSTM and GRU
"""
import torch
import torch.nn as nn
from utils import Timer

# Evaluates model on validation dataset 
# Returns average accuracy
def evaluate(model, data_loader):
    correct_total = 0
    outputs_total = 0
    for inputs, targets in data_loader:
        outputs = model(inputs)
        outputs_total += targets.numel()
        correct_total += torch.sum(outputs.argmax(dim=2) == targets)
    return correct_total / outputs_total


# A whole training process of given untrained model
def train(model, train_loader, dev_loader, num_epoches=10, log_interval=100, learning_rate=1e-3):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Starts training loop.
    print("Training starts...")
    Timer.Start(num_epoches * len(train_loader))
    for epoch in range(num_epoches):
        running_loss = 0
        for idx, batch in enumerate(train_loader):
            inputs, targets = batch
            outputs = model(inputs)

            # Shape of outputs (sentence_length, batch_size, output_size)
            # Shape of targets (sentence_length, batch_size)
            outputs = outputs.view((-1, model.output_size))
            targets = targets.view((-1))
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Occationaly logging some info
            running_loss += loss.item()
            if idx % log_interval == log_interval - 1:
                print("Epoch: %d\tStep: %d\tLoss= %.3f\tAcc: %.3f\t%s" % (
                    epoch + 1,
                    idx,
                    running_loss / log_interval,
                    evaluate(model, dev_loader),
                    Timer.Remain(epoch * len(train_loader) + idx)
                ))     
                running_loss = 0
    print("Training finishes!")
    Timer.Stop()
