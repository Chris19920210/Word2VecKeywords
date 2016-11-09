第一步安装python dependency packages
请在resource文件中找到requirements.txt文件
sudo pip install -r requirements.txt

第二步（可选）：
将文章分割成句子，句子分词，制作成训练样本
python dbLoaderHtmlparser.py --host 172.16.210.83 --user news --passwd news --db news_db

--host: mysql host
--user: mysql user
--passwd: mysql password
--db: t_news_detail 所在数据库

程序运行后会在resource文件夹下生成split_words.txt文件

第三步（可选）
将上一步生成的split_word.p训练集放入doc2vec模型中进行训练
python model_training.py --input ..resource/split_words.txt --size 400 --window 300 --mincount 3 --workers 4

--input: 为split_words.txt的路径
--size: 为句子向量表示的长度。是模型的hyperparameter，可根据模型表现进行调整
--window: 为中心词两边考虑窗口大小，是模型的hyperparameter，可根据模型表现进行调整
--mincount: 为最短考虑的句子，若句子分词后小于mincount则从训练集中删除，是模型的hyperparameter，可根据模型表现进行调整
--workers: 模型训练所需的线程数

程序运行后会在当前目录下生成Doc2vec*等一系列文件，是将句子向量化的模型

第四步 webservice
python web_textrank_2.py
url: http://127.0.0.1:5000/todos
params:

content: 文章内容
keyword:文章分类的标签


由于已经根据数据库中的语料生成了模型，所以可直接利用第四步对新存入数据库中的文章进行摘要及关键词抽取。



文件夹中的文档均有用请不要删除

stop_words.txt - 分词过程中需要忽略的词语
punc_file.txt - 分词前需要过滤的符号等冗余信息
redundant_dict.txt - 包含一些关键词，textrank算法将不会输出包含这些词的句子
userdict.txt - 分词时可处理为连词的词

src文件夹下为代码
resource为训练好的模型，在启动webservice的时候需要加载。包括doc2vec模型和word2vec模型
data为可测试的数据样本keyword = 人工智能
doc中为readme

