__author__ = 'SherlockLiao'

import os
import torch


class Dictionary(object):
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}

    def add_word(self, word_list):
        for word in word_list:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
                self.idx_to_word[len(self.word_to_idx)-1] = word

    def __len__(self):
        return len(self.word_to_idx)


class Corpus(object):
    def __init__(self, path='./data'):
        self.dic = Dictionary()
        self.train = os.path.join(path, 'train.txt')
        self.valid = os.path.join(path, 'valid.txt')
        self.test = os.path.join(path, 'test.txt')
        self.path = path

    def get_data(self, file, batch_size=20):
        file = os.path.join(self.path, file)
        # get the word dictionary
        with open(file, 'r') as f:
            num_word = 0
            for line in f:
                word_list = line.split() + ['<eos>']
                num_word += len(word_list)
                self.dic.add_word(word_list)

        token = torch.LongTensor(num_word)
        # get the whole sentence corpus
        with open(file, 'r') as f:
            index = 0
            for line in f:
                word_list = line.split() + ['<eos>']
                for word in word_list:
                    token[index] = self.dic.word_to_idx[word]
                    index += 1
        num_batch = index // batch_size
        token = token[: num_batch*batch_size]
        token = token.view(batch_size, -1)
        return token
