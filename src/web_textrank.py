#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import jieba
from util import cut_sentence_new
from util import data_generators
from jieba import analyse
from collections import namedtuple
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx
from util import rankgetter
from util import distattr
from util import string_parser
from gensim.models import doc2vec
from util import contain_redundant
from util import removePrefix
from util import is_chinese
from util import distattr2
import codecs
import soaplib
# from soaplib.core.util.wsgi_wrapper import run_twisted #发布服务
from soaplib.core.server import wsgi
from soaplib.core.service import DefinitionBase  #所有服务类必须继承该类
from soaplib.core.service import soap  #声明注解
from soaplib.core.model.clazz import Array #声明要使用的类型
from soaplib.core.model.clazz import ClassModel  #若服务返回类，该返回类必须是该类的子类
from soaplib.core.model.primitive import Integer,String

class AbstractExtraction(DefinitionBase):
    # load model as attribute
    model = doc2vec.Doc2Vec.load("../resource/Doc2vecModel")

    def matrix_getter(self, dicts):
        X_tuple = sorted(dicts.items(), key = lambda (key, value): key)
        X = np.array([each[1] for each in X_tuple], dtype = 'float32')
        return X


    def sentence_evaluate(self, sentence):
        seg_list = jieba.cut(string_parser(punc_file = '../resource/punc_file.txt',string_with_punc = sentence))
        seg_list = filter(lambda x: x != " ", seg_list)
        lists = list(seg_list)
        return lists

    def stdOut(self, rank,dicts, top):
        lists = list()
        j = 0
        try:
            for sentence_id in rank:
                tmp = dicts[sentence_id]
                tmp2 = filter(lambda x: is_chinese(x), tmp)
                if (len(tmp2) < 8 or contain_redundant(redundant_dict = '../resource/redundant_dict.txt', string_with_redundant = tmp)):
                    continue
                j += 1
                result_str = removePrefix(tmp.replace(" ",""),"”".decode("utf8"))
                result = distattr2(sentence_id, result_str)
                lists.append(result)
                if (j >=top):
                    break
            std = sorted(lists,key =lambda x: x.ids)
        except:
            std = lists
        return std


    # return score for each sentence
    @soap(String,Integer,_returns=Array(String))
    def evaluation(self,path,top):
        with codecs.open(path, 'rb','utf8') as f:
            content_html = f.read()
        content_tmp = BeautifulSoup(content_html,"html.parser")
        content = content_tmp.get_text()
        # split text to sentence
        lists = cut_sentence_new(content)
        dictssentence = {key: value for key, value in enumerate(lists)}
        # split sentence to words only consider the sentence with len greater than 6
        dictsword = {key: self.sentence_evaluate(value) for key, value in dictssentence.items() if len(self.sentence_evaluate(value)) > 6}
        # words to vector
        dictsvector = {key: self.model.infer_vector(value) for key, value in dictsword.items()}
        # matrix
        X = self.matrix_getter(dictsvector)
        # get distance matrix based on cosine distance
        distance_matrix = pairwise_distances(X,metric = 'cosine')
        nx_graph = nx.from_numpy_matrix(distance_matrix)
        scores = nx.pagerank(nx_graph)
        rank_tmp = sorted(scores.items(), key=lambda (key, value): value, reverse =True)
        rank = [each[0] for each in rank_tmp]
        result = self.stdOut(rank,dictssentence,top)
        output = [str(each.ids) + ": " + each.strs for each in result]
        return output










if __name__=='__main__':
    try:
        from wsgiref.simple_server import make_server
        soap_application = soaplib.core.Application([AbstractExtraction], 'tns')
        wsgi_application = wsgi.Application(soap_application)
        server = make_server('localhost', 7792, wsgi_application)
        server.serve_forever()
    except ImportError:
        print "Error: example server code requires Python >= 2.5"
