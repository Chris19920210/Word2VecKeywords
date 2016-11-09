#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jieba
from collections import namedtuple
import networkx as nx
import numpy as np

def cut_sentence_new(sentences):
        punt_list = '!?;~。！？；～'.decode('utf8') + "\n" + "\r"
        words = sentences.replace('\t','').replace(u"\xa0","").replace(u"\u3000","").replace("\r", "").strip(punt_list)
        start = 0
        i = 0
        sents = []
        for word in words:
            if word in punt_list and token1 not in punt_list:
                st = words[start:i+1]
                sents.append(st)
                start = i+1
                i += 1
            else:
                i += 1
                token1 = list(words[start:i+2]).pop()# 取下一个字符
        if start < len(words):
            st = words[start:] #去除标点
            sents.append(st)
        return sents


def cut_sentence(sentence):
    lists = []
    for line in sentences:
        seg_list = jieba.cut(line,cut_all=False)
        lists.append(list(seg_list))
    return lists

class distattr:
    def __init__(self, ids, index, score):
        self.ids = ids
        self.index = index
        self.score = score
    def __repr__(self):
        return repr((self.ids, self.index, self.score))

def rankgetter(**kwargs):
    dictrember = kwargs['dictrember']
    nx_graph = nx.from_numpy_matrix(kwargs['distance_matrix'])
    scores = nx.pagerank(nx_graph)
    result = list()
    for key, value in scores.items():
        tmp = distattr(dictrember[key], key, value)
        result.append(tmp)
    return sorted(result, key = lambda distattr: distattr.score, reverse = True)

def string_parser(**kwargs):
    string_with_punc = kwargs['string_with_punc']
    lists = list()
    with open(kwargs['punc_file'],'rb') as f:
        for punc in f.readlines():
            lists.append((punc.strip('\n').decode('utf8'),""))
    string_tmp = reduce(lambda a,kv: a.replace(*kv),lists,string_with_punc)
    bare_string = string_tmp.strip('”“.!?:;~。！？：；～'.decode('utf8') + "\n")
    return bare_string

# filter out sentence with redundant info
def contain_redundant(**kwargs):
    lists = list()
    string_with_redundant = kwargs['string_with_redundant']
    with open(kwargs['redundant_dict'],'rb') as f:
        for punc in f.readlines():
            lists.append(punc.strip('\n').decode('utf8'))
    for redun in lists:
        flag = False
        if redun in string_with_redundant:
            flag = True
            break
    return flag

def removePrefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text



def is_chinese(uchar):

        """判断一个unicode是否是汉字"""

        if uchar >= u'\u4e00' and uchar<=u'\u9fa5':

                return True

        else:

                return False


class distattr2:
    def __init__(self,ids, strs):
        self.ids = ids
        self.strs = strs
    def __repr__(self):
        return repr((self.ids, self.strs))


