__author__ = 'Jmexe'
#encoding=utf-8
import MySQLdb
import jieba
from gensim import corpora, models

def prepare(num):
    #load data
    db = MySQLdb.connect("129.63.16.112","root","wtsql2014","KuaiwenbaoDB", charset="utf8")
    cursor = db.cursor()
    sql = "select * from KuaiwenbaoDB.xinwendb_1503 limit " + str(num)

    cursor.execute(sql)
    data = cursor.fetchall()

    #tokenize
    documents = []

    stopkey=[line.strip().decode('utf-8') for line in open('./stop_words_list.txt').readlines()]

    for doc in data:
        text = doc[2]
        documents.append(list(set(jieba.cut(text)) - set(stopkey)))


    return documents

    """
    #TODO generate dictionary
    dictionary = corpora.Dictionary(documents)
    dictionary.save("./test.dict")

    #TODO generate corpus
    corpus = [dictionary.doc2bow(text) for text in documents]
    corpora.MmCorpus.serialize('./test.mm', corpus)
    """
