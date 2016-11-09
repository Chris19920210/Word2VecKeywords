#!/usr/bin/env python
# -*- coding: utf-8 -*-
import MySQLdb
import MySQLdb.cursors
import argparse
from util import data_generators
import pickle
from jieba import analyse
from collections import namedtuple
from gensim.models import doc2vec
from collections import namedtuple
from gensim.models import Work2Vec
parser = argparse.ArgumentParser(description='train doc2vec model')

parser.add_argument('--input', type=str, default = './split_word.p',
                    help='input for training')

parser.add_argument('--size', type=int, default = 200,
                    help='size for representation vector')

parser.add_argument('--window', type=int, default = 200,
                    help='window for context')
parser.add_argument('--mincount', type=int,default = 3,
                    help='threshould for vectorization')
parser.add_argument('--workers', type=int,default = 4,
                    help= 'num of threads for training')
args = parser.parse_args()

def main():
    # load data for training
    with open(args.input, 'rb') as f:
        dicts = pickle.load(f)
    # map dicts to lists
    docs = map(lambda x: x[1], dicts.items())
    # training the model
    model = Word2Vec(docs, size = args.size, window = args.window, min_count = args.mincount, workers = args.workers)
    return model

if __name__ == '__main__':
    model = main()
    model.save('Word2vecModel')
    print('Word2VecModel Done!')




