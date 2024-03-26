import math
import numpy as np
import logging
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from preprocess import split_into_sentence,delete_stop
import json
from tqdm import tqdm
import logging
import copy
import re
from preprocess import delete_num
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# 测试文本
test1_ent = ['liquigas-cannondale','astana','team sky','stage 18 result','commemorated','anniversary']
#test_ent = ['nonso anozie', 'robert','series','character']
test_ent = ["jonathan roberts", "celebrity", "dancing with the stars", "season 5"]


def evaluate_retriever(search_docs,docs,test_path,out_path,recallk:int=20):
    with open(test_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    max_len = len(questions)
    correct = 0
    generation_dict = dict()
    print("\n=========Start counting sim============\n")
    for id in tqdm(questions):
        acc ="false"
        match_ent = questions[id]["matchwords"]
        #是否去重？这是个问题
        #test_ent = list(set(match_ent))
        index, table_id, top_arr = search_BM25(search_docs, docs, match_ent, topk=recallk)
        # print(top_arr)
        # print(index)
        print(table_id)
        ground_id = questions[id]["data_item"]["table_id"]
        if ground_id in table_id:
            acc = "true"
            print("查到")
            correct = correct+1
        generation_dict[id]={
            "search_id":table_id,
            "acc": acc,
            'data_item': copy.deepcopy(questions[id]["data_item"])
        }
    with open(out_path, 'w', encoding='utf-8') as fw:
        json.dump(generation_dict, fw, indent=4,ensure_ascii=False)

    accuracy = correct/max_len
    print("recall{}情况下的搜索正确率为：{}".format(recallk,accuracy))


class BM25(object):
  def __init__(self,docs):
    self.docs = docs   # 传入的docs要求是已经分好词的list
    self.doc_num = len(docs) # 文档数
    self.vocab = set([word for doc in self.docs for word in doc]) # 文档中所包含的所有词语
    self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.doc_num # 所有文档的平均长度
    self.k1 = 0.2
    self.b = 0.4

  def idf(self,word):
    if word not in self.vocab:
      word_idf = 0
    else:
      qn = {}
      for doc in self.docs:
        if word in doc:
          if word in qn:
            qn[word] += 1
          else:
            qn[word] = 1
        else:
          qn[word] = 0
          continue
      word_idf = np.log((self.doc_num - qn[word] + 0.5) / (qn[word] + 0.5))
    return word_idf

  def score(self,word):
    score_list = []
    idf_word = self.idf(word)
    for index,doc in enumerate(self.docs):
      word_count = Counter(doc)
      if word in word_count.keys():
        f = word_count[word]+0.0
      else:
        f = 0.0
      r_score = (f*(self.k1+1)) / (f+self.k1*(1-self.b+self.b*len(doc)/self.avgdl))
      score_list.append(idf_word * r_score)
      #score_list.append(r_score)
    return score_list

  def score_all(self,sequence):
    sum_score = []
    for word in sequence:
      sum_score.append(self.score(word))
    sim = np.sum(sum_score,axis=0)
    return sim


def extract_ent(all_path,docs_path):
    '''
    build the set of entitles
    '''
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
    punc_dicts = [i for i in punctuation]
    stopWords = set(stopwords.words('english'))
    with open(all_path, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    set_tables=[]
    logger.info("start split and extract entitles")
    for id,table in tqdm(tables.items()):
        set_ent = []
        set_info = []
        # for tit in table["header"]:
        #     set_ent.append(tit.lower())
        # for ent in table["data"]:
        #     for en in ent:
        #         set_ent.append(en.lower())
        ori_tit = table["title"].lower().strip()
        set_ent.append(ori_tit)
        set_ent.append(ori_tit)
        de_tit = delete_num(ori_tit)
        if de_tit != ori_tit:
            set_ent.append(de_tit)
        # if table["section_title"] != "":
        #     set_ent.append(table["section_title"].lower())
        # if table["section_text"] != "":
        #     set_ent.append(table["section_text"].lower())
        for sent in split_into_sentence(table["intro"]):
            set_info.append(sent.lower())
            # sent_words = word_tokenize(sent.lower())
            # for w in sent_words:
            #     if w.lower() not in stopWords and w.lower() not in punc_dicts:
            #         set_info.append(w)
        set_table = set_ent
        set_tables.append({
            id : set_table
        })
    with open(docs_path, 'w', encoding='utf-8') as fw:
        json.dump(set_tables, fw, indent = 4,ensure_ascii=False)
    return set_tables

def search_BM25(search_docs,docs,query,topk: int = 10):
    #改bug：之前运行很慢，后来发现是因为代码里每次循环doc都算了一遍idf。
    bm = BM25(search_docs)
    score = bm.score_all(query)
    arr = np.array(score)
    top_arr = np.sort(-arr)[:topk]
    # print(np.argsort(-arr))
    # print(score)
    index = np.argsort(-arr)[:topk]
    table_id = []
    kvpairs = list(docs)
    for i in index:
        a = list(kvpairs[i])
        table_id.append(list(kvpairs[i])[0])
    return index,table_id,top_arr

def search_docs(docs):
    set_tables = []
    for table in docs:
        cells = list(table.values())[0]
        set_tables.append(cells)
    return set_tables

def one_match(search_dic,match_ent):
    table_id = []
    for table in search_dic:
        cells = list(table.values())[0]
        id = list(table.keys())[0]
        for cell in cells:
            if cell in match_ent:
                table_id.append(id)
                break
    return table_id


def match_all(docs,keyword_path,out_path):
    with open(keyword_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    max_len = len(questions)
    correct = 0
    generation_dict = dict()
    table_id = []
    for id in tqdm(questions):
        acc ="false"
        match_ent = questions[id]["matchwords"]
        key_ent = questions[id]["keywords"][:-1]
        #是否去重？这是个问题
        test_ent = list(set(match_ent))
        table_id = one_match(docs,test_ent)
        #print("第{}个问题找到的表个数为{}".format(id,len(table_id)))
        ground_id = questions[id]["data_item"]["table_id"]
        if ground_id in table_id:
            acc = "true"
            #print("查到")
            correct = correct+1
        if ground_id not in table_id and mode == "NQ" :
            table_id = match_dev(key_ent,"./data/NQ/BM25/test_cell.json")
            if ground_id in table_id:
                acc = "true"
                correct = correct + 1
        generation_dict[id] = {
            "search_id": table_id,
            "acc": acc,
            'data_item': copy.deepcopy(questions[id]["data_item"])
        }

    with open(out_path, 'w', encoding='utf-8') as fw:
        json.dump(generation_dict, fw, indent=4,ensure_ascii=False)
    accuracy = correct/max_len
    print("简单搜索情况下的搜索正确率为：{}".format(accuracy))

def extract_onetable(tables,s_id):
    table = tables[s_id]
    set_h = []
    set_e = []
    for tit in table["header"]:
        set_h += tit.lower().split(" ")
    for ent in table["data"]:
        for en in ent:
            set_e += en.lower().split(" ")
    set_t = table["title"].lower().strip().split(" ")
    set_u = table["uid"].lower().strip().split("_")
    set_all = set_h+set_e+set_t+set_u
    return set_all



def reranker(sim_path,out_path):
    with open(sim_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    with open("./data/OTT-QA/all_plain_tables.json", 'r', encoding='utf-8') as f1:
        tables = json.load(f1)
    generation_dict = dict()
    print("--------------start reranker----------------------")
    for id in tqdm(questions):
        set_docs = []
        bm25_match = []
        if questions[id]["acc"]== "true":
            clean_text = re.sub(r'[^\w\s]', '', questions[id]["data_item"]["question"])
            ques_word = clean_text.lower().strip().split(" ")
            ques_word = [q for q in ques_word if q !=""]
            fil_quesword = delete_stop(ques_word)
            for s_id in questions[id]["search_id"]:
                set_ent = extract_onetable(tables,s_id)
                set_docs.append(set_ent)
            bm = BM25(set_docs)
            score = bm.score_all(fil_quesword)
            arr = np.array(score)
            index = np.argsort(-arr)
            for i in index:
                bm25_match.append(questions[id]["search_id"][i])
            generation_dict[id]={
                "search_id": bm25_match,
                "acc": "true",
                "data_item":copy.deepcopy(questions[id]["data_item"])
            }
        else:
            generation_dict[id]={
                "search_id": [],
                "acc": "false",
                "data_item":copy.deepcopy(questions[id]["data_item"])
            }
    with open(out_path, 'w', encoding='utf-8') as fw:
        json.dump(generation_dict, fw, indent=4,ensure_ascii=False)


def nq_reranker(sim_path,out_path):
    with open(sim_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    with open("./data/NQ/tables/tables.jsonl", 'r', encoding='utf-8') as f1:
        tables = json.load(f1)
    generation_dict = dict()
    print("--------------start reranker----------------------")
    for id in tqdm(questions):
        set_docs = []
        bm25_match = []
        if questions[id]["acc"]== "true":
            clean_text = re.sub(r'[^\w\s]', '', questions[id]["data_item"]["question"])
            ques_word = clean_text.lower().strip().split(" ")
            ques_word = [q for q in ques_word if q !=""]
            fil_quesword = delete_stop(ques_word)
            for s_id in questions[id]["search_id"]:
                set_ent = extract_onetable(tables,s_id)
                set_docs.append(set_ent)
            bm = BM25(set_docs)
            score = bm.score_all(fil_quesword)
            arr = np.array(score)
            index = np.argsort(-arr)
            for i in index:
                bm25_match.append(questions[id]["search_id"][i])
            generation_dict[id]={
                "search_id": bm25_match,
                "acc": "true",
                "data_item":copy.deepcopy(questions[id]["data_item"])
            }
        else:
            generation_dict[id]={
                "search_id": [],
                "acc": "false",
                "data_item":copy.deepcopy(questions[id]["data_item"])
            }
    with open(out_path, 'w', encoding='utf-8') as fw:
        json.dump(generation_dict, fw, indent=4,ensure_ascii=False)

def extract_nq_ent(all_path,docs_path):
    '''
    build the set of entitles
    '''
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
    punc_dicts = [i for i in punctuation]
    stopWords = set(stopwords.words('english'))
    set_tables=[]
    logger.info("start split and extract entitles")
    with open(all_path, 'r', encoding='utf-8') as f:
        for line in f:
            set_ent = []
            table = json.loads(line)
            title_id = table["tableId"]
            ori_tit = table["documentTitle"].lower().strip()
            set_ent.append(ori_tit)
            set_ent.append(ori_tit)
            de_tit = delete_num(ori_tit)
            if de_tit != ori_tit:
                set_ent.append(de_tit)
            set_tables.append({
                title_id: set_ent
            })

    with open(docs_path, 'w', encoding='utf-8') as fw:
        json.dump(set_tables, fw, indent = 4,ensure_ascii=False)
    return set_tables

def not_number(num):
    try:
        int(num)
        return False
    except ValueError:
        return True

def extract_nq_e(all_path,docs_path):
    '''
    build the set of entitles
    '''
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
    punc_dicts = [i for i in punctuation]
    stopWords = set(stopwords.words('english'))
    set_tables=[]
    logger.info("start split and extract entitles")
    with open(all_path, 'r', encoding='utf-8') as f:
        for line in f:
            set_ent = []
            table = json.loads(line)
            title_id = table["table"]["tableId"]
            ori_tit = table["table"]["documentTitle"].lower().strip()
            set_ent.append(ori_tit)
            set_ent.append(ori_tit)
            for col in table["table"]["columns"]:
                col_text = col["text"]
                if col_text != "" and not_number(col_text) :
                    set_ent.append(col_text.lower().strip())
            for row in table["table"]["rows"]:
                for row_cell in row["cells"]:
                    cell_text = row_cell["text"]
                    if cell_text != "" and not_number(cell_text) :
                        set_ent.append(cell_text.lower().strip())

            de_tit = delete_num(ori_tit)
            if de_tit != ori_tit:
                set_ent.append(de_tit)
            set_tables.append({
                title_id: set_ent
            })

    with open(docs_path, 'w', encoding='utf-8') as fw:
        json.dump(set_tables, fw, indent = 4,ensure_ascii=False)
    return set_tables

def match_dev(keyword,cell_doc):
    with open(cell_doc, 'r', encoding='utf-8') as f:
        cells = json.load(f)

    all_score = []
    for table in cells:
        t_score = 0
        for t_id in table:
            cell = table[t_id]
            for ce in cell:
                for iden in keyword:
                    set_iden = iden.split(" ")
                    if len(set_iden) > 1:
                        if iden in ce:
                            t_score += 1
                    else:
                        if iden == ce :
                            t_score += 1

        all_score.append(t_score)
    arr = np.array(all_score)
    index = np.argsort(-arr)[:50]
    match_id = []
    for i in index:
        if all_score[i] != 0:
            m_id = list(cells[i].keys())[0]
            match_id.append(m_id)
    return match_id

def extract_nq_onetable(tables,s_id):
    table = tables[s_id]
    set_h = []
    set_e = []
    for tit in table["columns"]:
        if tit["text"] != "":
            set_h += tit["text"].lower().split(" ")
    for row in table["rows"]:
        for en in row["cells"]:
            if en["text"] != "":
                set_e += en["text"].lower().split(" ")
    set_t = table["documentTitle"].lower().strip().split(" ")
    set_all = set_h+set_e+set_t+set_t
    return set_all

def nq_reranker(sim_path,out_path):
    with open(sim_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    all_table_dict = dict()
    with open("./data/NQ/tables/tables.jsonl", 'r', encoding='utf-8') as f1:
        for line in f1:
            data = json.loads(line)
            t_id = data["tableId"]
            all_table_dict[t_id] = data
    generation_dict = dict()
    print("--------------start reranker----------------------")
    for id in tqdm(questions):
        set_docs = []
        bm25_match = []
        if questions[id]["acc"]== "true":
            clean_text = re.sub(r'[^\w\s]', '', questions[id]["data_item"]["question"])
            ques_word = clean_text.lower().strip().split(" ")
            ques_word = [q for q in ques_word if q !=""]
            fil_quesword = delete_stop(ques_word)
            for s_id in questions[id]["search_id"]:
                set_ent = extract_nq_onetable(all_table_dict,s_id)
                set_docs.append(set_ent)
            bm = BM25(set_docs)
            score = bm.score_all(fil_quesword)
            arr = np.array(score)
            index = np.argsort(-arr)
            for i in index:
                bm25_match.append(questions[id]["search_id"][i])
            generation_dict[id]={
                "search_id": bm25_match,
                "acc": "true",
                "data_item":copy.deepcopy(questions[id]["data_item"])
            }
        else:
            generation_dict[id]={
                "search_id": [],
                "acc": "false",
                "data_item":copy.deepcopy(questions[id]["data_item"])
            }
    with open(out_path, 'w', encoding='utf-8') as fw:
        json.dump(generation_dict, fw, indent=4,ensure_ascii=False)


def evaluate_topk(search_path,topk):
    with open(search_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    max_len = len(questions)
    correct = 0
    for id in tqdm(questions):
        table_id = questions[id]["search_id"]
        if len(table_id) > topk:
            table_id = table_id[:topk]
        ground_id = questions[id]["data_item"]["table_id"]
        if ground_id in table_id:
            correct = correct + 1
    accuracy = correct/max_len
    print("recall{}情况下的搜索正确率为：{}".format(topk,accuracy))

if __name__ == "__main__":
    #docs_path = "./data/NQ/BM25/title_cell.json"
    docs_path = "./data/OTT-QA/BM25/title_cell.json"
    #docs_path = None
    if docs_path == None:
        #docs = extract_ent("./data/OTT-QA/all_plain_tables.json","./data/OTT-QA/BM25/title_cell.json")
        docs = extract_nq_ent("./data/NQ/tables/tables.jsonl", "./data/NQ/BM25/title_cell.json")
    else:
        with open(docs_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
    print(len(docs))

    #extract_nq_e("./data/NQ/interactions/test.jsonl", "./data/NQ/BM25/test_cell.json")
    # match_id = match_dev(["seasons","one tree hill"] , "./data/NQ/BM25/dev_cell.json")
    # print(match_id)

    mode = "OTT-QA"
    if mode == "NQ":
        match_all(docs,"./data/NQ/BM25/faiss_keyword_nqtables_test.json","./data/NQ/BM25/simsearch_all_nqtables_test.json")
        # #bm25重新排序
        nq_reranker("./data/NQ/BM25/simsearch_all_nqtables_test.json","./data/NQ/BM25/reranker_nqtables_test.json")
        # # #表格搜索评分
        evaluate_topk("./data/NQ/BM25/reranker_nqtables_test.json",topk=50)

    if mode == "OTT-QA":
        #match_all(docs,"./data/OTT-QA/BM25/faiss_one_ottqa_dev.json","./data/OTT-QA/BM25/simsearch_all_ottqa_devone.json")
        #bm25重新排序
        #reranker("./data/OTT-QA/BM25/simsearch_all_ottqa_devone.json","./data/OTT-QA/BM25/reranker_ottqa_devone.json")
        #表格搜索评分
        evaluate_topk("./data/OTT-QA/BM25/reranker_ottqa_devone.json",topk=20)


    #
    # search_docs = search_docs(docs)
    # evaluate_retriever(search_docs,docs,"./data/OTT-QA/BM25/faiss_keyword_ottqa_test.json","./data/OTT-QA/BM25/BM25_top20_ottqa_test.json",recallk=20)

