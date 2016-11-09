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

app = Flask(__name__)
api = Api(app)
jieba.load_userdict("../resource/userdict.txt")
model_abstract = doc2vec.Doc2Vec.load("../resource/Doc2vecModel3")
model_word = Word2Vec.load("../resource/wiki.zh.text.model")
stopw = [line.strip().decode('utf-8') for line in open('../resource/stop_words.txt').readlines()]
TODOS = {}

parser = reqparse.RequestParser()
parser.add_argument('content',required=True)
parser.add_argument('keyword')

def abort_if_todo_doesnt_exist(todo_id):
    if todo_id not in TODOS:
        abort(404, message="Todo {} doesn't exist".format(todo_id))

def matrix_getter(disattr_list):
    X = np.array([each.strs for each in disattr_list], dtype = 'float32')
    return X

# only keep nouns
def word_evaluate(sentence):
    seg_list = pseg.cut(string_parser(punc_file = '../resource/punc_file.txt',string_with_punc = sentence))
    seg_list = filter(lambda x: x.word != "" and x.word not in stopw and 'n' in x.flag and len(x.word) >= 2, seg_list)
    words = map(lambda x: x.word, seg_list)
    words = list(set(words))
    return words


def sentence_evaluate(sentence):
    seg_list = pseg.cut(string_parser(punc_file = '../resource/punc_file.txt',string_with_punc = sentence))
    seg_list = filter(lambda x: x.word != "" and x.word not in stopw and 'n' in x.flag and len(x.word) >= 2, seg_list)
    words = map(lambda x: x.word, seg_list)
    words = filter(lambda x: len(x) != 0, words)
    return words

def stdOut(sentence_result_lists,dicts, scope):
    lists = list()
    j = 0
    sentence_nbr = len(dicts)
    sentence_result_lists_tmp = map(lambda (key, score): (key, score*(1 - math.log((key+1))/math.log(sentence_nbr))),sentence_result_lists)
    sentence_result_lists2 = sorted(sentence_result_lists_tmp, key = lambda (key, score): score, reverse = True)
    try:
        for distattr3 in sentence_result_lists2:
            sentence_id = distattr3[0]
            tmp = dicts[sentence_id]
            tmp2 = filter(lambda x: is_chinese(x), tmp)
            if (len(tmp2) < 8 or contain_redundant(redundant_dict = '../resource/redundant_dict.txt', string_with_redundant = tmp)):
                continue
            j += 1
            result_str = removePrefix(tmp.strip(" "),"”".decode("utf8"))
            result = distattr2(sentence_id, result_str)
            lists.append(result)
            if (j >=scope):
                break
        std = sorted(lists,key =lambda x: 0.5*len(x.strs)/(x.ids + 1), reverse = True)
    except:
        std = lists
    return std

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

def evaluate_sentence(dictssentence):
    dictsword = {key: sentence_evaluate(value) for key,value in dictssentence.items() if len(sentence_evaluate(value)) > 6}
    if len(dictsword) == 0:
        result_list_final = []
        return result_list_final
    else:
        sentence_vector_dicts = [distattr2(key,model_abstract.infer_vector(value)) for key, value in dictsword.items()]
        try:
            result_list = textrankgetter(sentence_vector_dicts)
            result_list_final = map(lambda x: (x.ids, x.score), result_list)
        except:
            result_list_final = []
        return  result_list_final

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


def evaluate():
    args = parser.parse_args()
    scope = 6
    content = args.content
    keyword = [args.keyword]
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
    # for sentence
    sentence_result_lists = evaluate_sentence(dictssentence)
    if len(sentence_result_lists) == 0:
        result_final = evaluate_sentence_tradition(dictssentence,words_importance)
        outputString = dictssentence.get(result_final[0].ids,"")
        return (outputString,words)
    else:
        sentence = stdOut(sentence_result_lists, dictssentence,6)
        if len(sentence) == 0:
            result_final = evaluate_sentence_tradition(dictssentence,words_importance)
            outputString = dictssentence.get(result_final[0].ids,"")
            return (outputString,words)
        else:
            outputString = sentence[0].strs
            return (outputString,words)


def evaluate_sentence_tradition(dictssentence,words_importance):
    result = list()
    words_lookup = dict(words_importance)
    sentence_nbr = len(dictssentence)
    for key, value in dictssentence.items():
        cut_sentence = sentence_evaluate(value)
        if len(cut_sentence) == 0:
            continue
        sentence_id = key
        importance_list = map(lambda x: words_lookup.get(x, 0),cut_sentence)
        score = reduce(lambda x,y: x + y, importance_list)*(1 - math.log((key+1))/math.log(sentence_nbr))
        result.append(distattr2(sentence_id, score))
    result_final = sorted(result,key = lambda x: 0.5*len(x.strs)/(sentence_id + 1),reverse =True)
    return result_final



# Todo
#   show a single todo item and lets you delete them
class Todo(Resource):

    def get(self, todo_id):
        abort_if_todo_doesnt_exist(todo_id)
        return TODOS[todo_id]

    def delete(self, todo_id):
        abort_if_todo_doesnt_exist(todo_id)
        del TODOS[todo_id]
        return '', 204

    def put(self, todo_id):
        tmp = evaluate()
        words_final = ",".join(tmp[1])
        TODOS[todo_id] = {'abstract': tmp[0],'keywords': words_final}
        return TODOS[todo_id], 201


class TodoList(Resource):
    def get(self):
        return TODOS
    def post(self):
        tmp = evaluate()
        words_final = ",".join(tmp[1])
        todo_id = len(TODOS) + 1
        TODOS[todo_id] = {'abstract': tmp[0],'keywords': words_final}
        return TODOS[todo_id], 201



##
## Actually setup the Api resource routing here
##
api.add_resource(TodoList, '/todos')
api.add_resource(Todo, '/todos/<todo_id>')


if __name__ == '__main__':
    import logging
    logFormatStr = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(format = logFormatStr, filename = "../log/global.log", level=logging.DEBUG)
    formatter = logging.Formatter(logFormatStr,'%m-%d %H:%M:%S')
    fileHandler = logging.FileHandler("../log/summary.log")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.DEBUG)
    streamHandler.setFormatter(formatter)
    app.logger.addHandler(fileHandler)
    app.logger.addHandler(streamHandler)
    app.logger.info("Logging is set up.")
    app.run(host='0.0.0.0', port=5000)
