import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader

from ipdb import set_trace
from sklearn import metrics

SEQ_LEN = 49
RNN_HID_SIZE = 64

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class TemporalDecay(nn.Module):
    def __init__(self, input_size):
        super(TemporalDecay, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(RNN_HID_SIZE, input_size))
        self.b = Parameter(torch.Tensor(RNN_HID_SIZE))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(35 * 2, RNN_HID_SIZE)

        self.regression = nn.Linear(RNN_HID_SIZE, 35)
        self.temp_decay = TemporalDecay(input_size = 35)

        self.out = nn.Linear(RNN_HID_SIZE, 1)

    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma = self.temp_decay(d)
            h = h * gamma
            x_h = self.regression(h)

            x_c =  m * x +  (1 - m) * x_h

            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            inputs = torch.cat([x_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(x_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        y_h = self.out(h)
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)

        # only use training labels
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        y_h = F.sigmoid(y_h)

        return {'loss': x_loss / SEQ_LEN + 0.1 *y_loss, 'predictions': y_h,\
                'imputations': imputations, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
