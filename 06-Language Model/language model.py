__author__ = 'SherlockLiao'

import torch
from torch.autograd import Variable
from torch import nn, optim
from data_utils import Corpus

seq_length = 30

train_file = 'train.txt'
valid_file = 'valid.txt'
test_file = 'test.txt'
train_corpus = Corpus()
valid_corpus = Corpus()
test_corpus = Corpus()

train_id = train_corpus.get_data(train_file)
valid_id = valid_corpus.get_data(valid_file)
test_id = test_corpus.get_data(test_file)

vocab_size = len(train_corpus.dic)
num_batches = train_id.size(1) // seq_length


class languagemodel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super(languagemodel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        x, hi = self.lstm(x, h)
        b, s, h = x.size()
        x = x.contiguous().view(b*s, h)
        x = self.linear(x)
        return x, hi


model = languagemodel(vocab_size, 128, 1024, 1)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def detach(states):
    return [Variable(state.data).cuda() for state in states]


for epoch in range(5):
    print('epoch {}'.format(epoch+1))
    print('*'*10)
    running_loss = 0
    states = (Variable(torch.zeros(1,
                                   20,
                                   1024)).cuda(),
              Variable(torch.zeros(1,
                                   20,
                                   1024)).cuda())

    for i in range(0, train_id.size(1)-2*seq_length, seq_length):
        input_x = train_id[:, i:(i+seq_length)]
        label = train_id[:, (i+seq_length):(i+2*seq_length)]
        if torch.cuda.is_available():
            input_x = Variable(input_x).cuda()
            label = Variable(label).cuda()
            label = label.view(label.size(0)*label.size(1), 1)
        else:
            input_x = Variabel(input_x)
            label = Variable(label)
            label = label.view(label.size(0)*label.size(1), 1)
        # forward
        states = detach(states)
        out, states = model(input_x, states)
        loss = criterion(out, label.view(-1))
        running_loss += loss.data[0]
        # backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        step = (i+1) // seq_length
        if step % 100 == 0:
            print('Epoch [{}/{}], Step[{}/{}], Loss: {}'
                  .format(epoch+1, 5, step, num_batches, loss.data[0]))
    print('Loss: {}'.format(running_loss))
