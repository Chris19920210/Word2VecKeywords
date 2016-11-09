#!/usr/bin/env python
# -*- coding: utf-8 -*-
import MySQLdb
import MySQLdb.cursors
from bs4 import BeautifulSoup
import argparse
import jieba
from util import cut_sentence_new
from util import data_generators
import pickle
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
#load data from mysql

parser = argparse.ArgumentParser(description='textrank train')

parser.add_argument('--host', type=str, default='localhost',
                    help='host for mysql')

parser.add_argument('--user', type=str, required = True,
                    help='user for mysql')

parser.add_argument('--passwd', type=str, required = True,
                    help='passwords for mysql')
parser.add_argument('--db', type=str,required =True,
                    help='database')
parser.add_argument('--file', nargs='+', type=str, default = ['1','1'], help='file for rating')
parser.add_argument('--top',type = int, default = 1, help= 'top sentences')

parser.add_argument('--model', type = str,default = '../resource/Doc2vecModel', help = 'trained model')

args = parser.parse_args()

def main():
    # load data
    conn = MySQLdb.connect(
                           host= args.host,
                           user=args.user,
                           passwd= args.passwd,
                           db= args.db,
                           charset='utf8')
    cur = conn.cursor()
    cur.execute('select id, content_html from t_crawler_obj limit ' + args.file[0] + ',' + args.file[1])
    data = cur.fetchall()

    # load model
    model = doc2vec.Doc2Vec.load(args.model)
    # parse data by beautiful soup
    dicts1 = dict()
    for line in data:
        ids, content_html = line
        content = BeautifulSoup(content_html,"html.parser")
        dicts1[ids] = content.get_text()

    # split sentence # nested dict dict2-> key: paper, value: dicttmp-> key: sentence id, value: sentence string
    dicts2 = defaultdict(dict)
    for key,value in dicts1.items():
        lists = cut_sentence_new(value)
        dicttmp = dict()
        for key2,value2 in enumerate(lists):
            dicttmp[key2] = value2
        dicts2[key] = dicttmp

# split words dict3-> key: paper, value: dicttmp-> key: sentence id, value: sentence split list
    dicts3 = defaultdict(dict)
    analyse.set_stop_words('../resource/stop_words.txt')
    for key,value in dicts2.items():
        dicttmp = dict()
        for key2,value2 in value.items():
            seg_list = jieba.cut(string_parser(punc_file = '../resource/punc_file.txt',string_with_punc = value2))
            seg_list = filter(lambda x: x != " ", seg_list)
            lists = list(seg_list)
            if(len(lists) >= 3): #save sentence with length greater than 3
                dicttmp[key2] = lists
        dicts3[key] = dicttmp

# vectorization and textrank

    for key, value in dicts3.items():
        dictrember = dict()
        X = list()
        i = 0
        for key2, value2 in value.items():
            dictrember[i] = key2 # i: X index; key2: sentence order
            X.append(model.infer_vector(value2))
            i += 1
        X = np.array(X, dtype = 'float32')
        distance_matrix = pairwise_distances(X,metric = 'cosine')
        rank = rankgetter(distance_matrix = distance_matrix, dictrember = dictrember)
        j = 0
        try:
            lists = list()
            for info in rank:
                ind = info.ids # sentence order
                tmp = dicts2[key][ind]
                tmp2 = filter(lambda x: is_chinese(x), tmp)
                if (len(tmp2) < 8 or contain_redundant(redundant_dict = '../resource/redundant_dict.txt', string_with_redundant = dicts2[key][ind])):
                    continue
                j += 1
                result_str = removePrefix(dicts2[key][ind].replace(" ",""),"”".decode("utf8"))
                result = distattr2(ind, result_str)
                lists.append(result)
                if (j >=args.top):
                    break

            stdOut = sorted(lists,key =lambda x: x.ids)# print the result according to the order sentence
            for key3, sentence3 in enumerate(stdOut):
                print str(key) +" "+ str(key3 + 1) +": " + sentence3.strs

        except:
            print("No More Qualified Sentence!")

if __name__ == '__main__':
    main()
    print("Abstract done!")
