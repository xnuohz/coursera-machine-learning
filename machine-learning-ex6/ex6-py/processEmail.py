# -*- coding:utf-8 -*-
import re
import nltk
import nltk.stem.porter
from getVocabList import getVocabList


def processEmail(email_contents):
    vocab = getVocabList()
    word_indices = []
    # Preprocess Email
    email_contents = email_contents.lower()
    # 去掉html标签
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    # 数字替换成number字符串
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    # 处理url链接格式
    email_contents = re.sub(
        '(http|https)://[^\s]*', 'httpaddr', email_contents)
    # 处理Email格式
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    # 替换$符号
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    # Tokenize Email

    print('==== Processed Email ====')
    stemmer = nltk.stem.porter.PorterStemmer()
    tokens = re.split('[@$/#.-:&*+=\[\]?!(){\},\'\">_<;% ]', email_contents)
    for token in tokens:
        # 除了字母和数字其他符号都去掉
        token = re.sub('[^a-zA-Z0-9]', '', token)
        token = stemmer.stem(token)
        if len(token) < 1:
            continue
        if token in vocab:
            word_indices.append(vocab[token])
    return word_indices
