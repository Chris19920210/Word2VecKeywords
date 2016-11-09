#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx
from util import string_parser
from gensim.models import doc2vec
from util import contain_redundant
from util import removePrefix
from util import is_chinese
from util import distattr2
from util import distattr
import codecs
from flask import Flask
from flask.ext.restful import reqparse, abort, Api, Resource
from bs4 import BeautifulSoup
from flask import request
from util import cut_sentence_new
from gensim.models import Word2Vec
import jieba
import urllib
from compiler.ast import flatten
from collections import Counter
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from jieba import analyse
import jieba.posseg as pseg
from sklearn.metrics.pairwise import cosine_similarity
import math
import os
import codecs

model_word = Word2Vec.load("../resource/wiki.zh.text.model")
jieba.load_userdict("../resource/userdict.txt")
stopw = [line.strip().decode('utf-8') for line in open('../resource/stop_words.txt').readlines()]

def matrix_getter(disattr_list):
    X = np.array([each.strs for each in disattr_list], dtype = 'float32')
    return X

def sentence_evaluate(sentence):
    seg_list = pseg.cut(string_parser(punc_file = '../resource/punc_file.txt',string_with_punc = sentence))
    seg_list = filter(lambda x: x.word != "" and x.word not in stopw and 'n' in x.flag and len(x.word) >= 2, seg_list)
    words = map(lambda x: x.word, seg_list)
    words = filter(lambda x: len(x) != 0, words)
    return words

def top50words(corpus):
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()  #所有文本的关键字
    weight = tfidf.toarray()
    word_importances = weight.mean(axis = 0)
    # sorted words by tfidf
    word2importances = sorted(zip(word,word_importances),key = lambda (word, importances): importances, reverse =True)
    return word2importances

def textrankgetter(disattr_list, sentence = True):
    X = matrix_getter(disattr_list)
    words = [each.ids for each in disattr_list]
    distance_matrix = cosine_similarity(X)
    nx_graph = nx.from_numpy_matrix(distance_matrix)
    if(sentence):
        scores = nx.pagerank(nx_graph,max_iter=1000,tol=5e-4)
    else:
        scores = nx.pagerank(nx_graph,max_iter=1000,tol=5e-3)
    scores_tuple = sorted(scores.items(), key = lambda (key,value): key)
    result_list_tmp = map(lambda (word,(index, score)): distattr(word,index,score),zip(words,scores_tuple))
    result_list = sorted(result_list_tmp, key=lambda x: x.score,reverse = True)
    return result_list

def word2vec_evaluate(word):
    try:
        vector = model_word[word]
    except:
        vector = np.array([])
    return vector

def evaluate_words(dictssentence,keyword,new=True):
    # for word !!!
    dictsword_tmp = {key: sentence_evaluate(value) for key, value in dictssentence.items() if len(sentence_evaluate(value)) > 0}
    dictsword_tmp2 = map(lambda z: z[1], dictsword_tmp.items())
    corpus = map(lambda z: " ".join(z), dictsword_tmp2)
    wordlists = top50words(corpus)
    # sentence lists
    if(new):
        if keyword[0] is None:
            words_final = map(lambda (word, importance): word, wordlists)[:100]
        else:
            dictsword_tmp3 = list(set(flatten(dictsword_tmp2)))
            word_potential = map(lambda t: (t,similar_check_higher(t, keyword)), flatten(dictsword_tmp3))
            words_2 = sorted(word_potential, key = lambda (word, score):score, reverse =True)
            words_3 = filter(lambda (key,score):score != -1,words_2)
            words_final = map(lambda (key, score): key, words_3)[:100]
        # word importance based on word2vec
        # ids:word index: order id score: vector
        word_vector_dicts = [distattr2(word, word2vec_evaluate(word)) for word in words_final if len(word2vec_evaluate(word)) != 0]
        try:
            final_list = map(lambda x: (x.ids,x.score),textrankgetter(word_vector_dicts, False))
            return final_list
        except:
            return wordlists
    else:
        return wordlists

def similar_check_higher(word, keywords):
    result_tmp = []
    for each in keywords:
        result1 = similar_check(word, each)
        result_tmp.append(result1)
    return max(result_tmp)


def similar_check(word, keyword):
    try:
        return model_word.similarity(word,keyword)
    except:
        return -1

def evaluate(content, keyword):
    scope = 6
    if keyword[0] is not None:
        try:
            model_word[keyword[0]]
        except:
            keyword = list(jieba.cut(string_parser(punc_file = '../resource/punc_file.txt',string_with_punc = keyword[0])))

    # split text to sentence
    lists = cut_sentence_new(content)
    dictssentence = {key: value.strip("\n") for key, value in enumerate(lists)}
    # split sentence to words only consider the sentence with len greater than 6
    # for word !!! top 5 words
    words_importance = evaluate_words(dictssentence,keyword)[:20]
    words = map(lambda (word, importances): word, words_importance)
    keywords_list = map(lambda x: word2vec_evaluate(x), words)
    keywords_list = filter(lambda x: len(x) != 0, keywords_list)
    denomiator = len(keywords_list)
    agg = reduce(lambda x,y : x + y, keywords_list)
    agg_final = map(lambda x: str(x),agg/denomiator)
    return agg_final

def main():
    filenames = os.listdir("../resource/content")
    with codecs.open("../resource/features.txt","wb","utf8") as z:
        for file in filenames:
            with codecs.open("../resource/content/" + file, "rb", "utf8") as f:
                string = f.read()
                content = string[string.find("\n") + 1:]
                keyword = string[:string.find("\n")]
                feature = evaluate(content,[keyword])
                z.write(file[:file.find(".")])
                z.write(",")
                z.write(keyword)
                z.write(",")
                z.write(",".join(feature))
                z.write("\n")

if __name__ == '__main__':
    main()
    print('Done!')






