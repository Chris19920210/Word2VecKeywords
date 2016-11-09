__author__ = 'huang'

import os
import logging
import sys
import jieba
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

if __name__=='__main__':
    jieba.load_userdict("../resource/userdict.txt")
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    if len(sys.argv) < 3:
        print(globals()['__doc__'] %locals())
        sys.exit(1)

    inp, outp = sys.argv[1:3]
    space = ' '

    output = open(outp, 'w')
    inp = codecs.open('wiki.cn.text.jian', 'r',"utf8")

    for line in inp.readlines():
        seg_list = jieba.cut(line)
        output.write(space.join(seg_list) + '\n')
