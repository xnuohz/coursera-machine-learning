# -*- coding:utf-8 -*-
import re


def getVocabList():
    vocab = {}
    for line in open('vocab.txt', 'r'):
        (val, key) = line.split()
        vocab[key] = int(val)
    return vocab


def getVocabListReverse():
    vocab = {}
    for line in open('vocab.txt', 'r'):
        (val, key) = line.split()
        vocab[int(val)] = key
    return vocab
