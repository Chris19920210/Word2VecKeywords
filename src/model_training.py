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
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
parser = argparse.ArgumentParser(description='train doc2vec model')

parser.add_argument('--input', type=str, default = '../resource/split_word.p',
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


def data_generators(data):
    return analyzedDocument(value, key)

def main():
    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    # load data for training
    with codecs.open(args.input, 'rb',"utf8") as f:
        for line in f.readlines():
            split_line = line.split(",",-1)
            key = split_line[0]
            value = filter(lambda x: x !=" ", split_line[1:])
            docs.append(analyzedDocument(value, key))

    # create labeled data
    # training the model
    model = doc2vec.Doc2Vec(docs, size = args.size, window = args.window, min_count = args.mincount, workers = args.workers)
    return model

if __name__ == '__main__':
    model = main()
    model.save('../resource/Doc2vecModel')
    print('Doc2VecModel Done!')




