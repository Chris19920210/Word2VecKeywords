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
from util import string_parser
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#load data from mysql

parser = argparse.ArgumentParser(description='load from mysql to split words')

parser.add_argument('--host', type=str, default='localhost',
                    help='host for mysql')

parser.add_argument('--user', type=str, required = True,
                    help='user for mysql')

parser.add_argument('--passwd', type=str, required = True,
                    help='passwords for mysql')
parser.add_argument('--db', type=str,required =True,
                    help='database')

args = parser.parse_args()

def main():
    conn = MySQLdb.connect(
                           host= args.host,
                           user=args.user,
                           passwd= args.passwd,
                           db= args.db,
                           charset='utf8')
    cur = conn.cursor()
    cur.execute('select id, content from t_news_detail order by rand() limit 200000')
    data = cur.fetchall()
# parse data by beautiful soup
    with open("../resource/split_words.txt", "wb") as f:
        for line in data:
            ids, content_html = line
            content = BeautifulSoup(content_html)
            text = content.get_text()
            try:
                lists = cut_sentence_new(text)
                for key2,value2 in enumerate(lists):
                    label = str(ids) + "_" + str(key2)
                    seg_list = jieba.cut(string_parser(punc_file = '../resource/punc_file.txt',string_with_punc = value2))
                    seg_list = filter(lambda x: x != " ", seg_list)
                    lists = list(seg_list)
                    #save sentence with length greater than 3
                    if(len(lists) >= 3):
                        vector = ",".join(lists)
                        f.write(label + ',' + vector + '\n')
            except:
                continue
    f.close()

if __name__ == '__main__':
    main()
    print("DataClean done!")
