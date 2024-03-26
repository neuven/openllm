import json
import re
import tqdm
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import nltk

# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def split_into_sentence(one_para_text: str, splited_puncs=None):
    if splited_puncs is None:
        splited_puncs = ['.', '?', '!','\n']
    splited_re_pattern = '[' + ''.join(splited_puncs) + ']'

    para = one_para_text
    sentences = re.split(splited_re_pattern, para)
    sentences = list(filter(lambda sent: len(sent) != 0, sentences))

    #避免切割后句子过长
    assert len(sentences) < 500

    return sentences

def extract_dev(all_path,index_path,dev_path):
    logger.info("start extract dev table ids")
    with open(index_path, 'r', encoding='utf-8') as f:
        table_ids = json.load(f)
        set_dev_id = []
        for dev_id in table_ids["dev"]:
            set_dev_id.append(dev_id)

    with open(all_path, 'r', encoding='utf-8') as f2:
        all_table_ids = json.load(f2)
        #delete_id = []
        a = len(all_table_ids)
        for id in tqdm.tqdm(list(all_table_ids.keys())):
            if id not in set_dev_id:
                all_table_ids.pop(id)
        dev_table_ids = all_table_ids

    with open(dev_path, 'w', encoding='utf-8') as f3:
        json.dump(dev_table_ids, f3, ensure_ascii=False)

def extract_traindev(all_path,index_path,traindev_path):
    logger.info("start extract dev table ids")
    with open(index_path, 'r', encoding='utf-8') as f:
        table_ids = json.load(f)
        set_traindev_id = []
        for dev_id in table_ids["dev"]:
            set_traindev_id.append(dev_id)
        for dev_id in table_ids["train"]:
            set_traindev_id.append(dev_id)

    with open(all_path, 'r', encoding='utf-8') as f2:
        all_table_ids = json.load(f2)
        #delete_id = []
        a = len(all_table_ids)
        for id in tqdm.tqdm(list(all_table_ids.keys())):
            if id not in set_traindev_id:
                all_table_ids.pop(id)
        traindev_table_ids = all_table_ids

    with open(traindev_path, 'w', encoding='utf-8') as f3:
        json.dump(traindev_table_ids, f3, ensure_ascii=False)


def extract_ent(all_path,ent_path):
    '''
    build the set of entitles
    '''
    with open(all_path, 'r', encoding='utf-8') as f:
        tables = json.load(f)
        set_title=[]
        set_entitle=[]
        set_info=[]
        logger.info("start split and extract entitles")
        for table in tqdm.tqdm(tables.values()):
            for tit in table["header"]:
                set_entitle.append(tit.lower())
            for ent in table["data"]:
                for en in ent:
                    set_entitle.append(en.lower())
            set_title.append(table["title"].lower())
            if table["section_title"] != "":
                set_title.append(table["section_title"].lower())
            if table["section_text"] != "":
                set_title.append(table["section_text"].lower())
            for sent in split_into_sentence(table["intro"]):
                set_info.append(sent.lower())
        #print(set_title)
        #print(len(set_entitle))
        #print(set_info)

        #去重
        set_entitle = list(set(set_entitle))
        set_title = list(set(set_title))
        set_info = list(set(set_info))
    with open(ent_path, 'w', encoding='utf-8') as fw:
        for ent in set_entitle:
            fw.write(ent)
            fw.write("\n")
        for tit in set_title:
            fw.write(tit)
            fw.write("\n")
        for info in set_info:
            fw.write(info)
            fw.write("\n")

def cut_by_ngram(sentence, min_n, max_n):
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
    punc_dicts = [i for i in punctuation]
    rst = []
    # 遍历切分长度的规划
    for length in range(min_n, min(len(sentence), max_n) + 1):
        # 依照此刻切分长度进行切分
        for idx in range(0, len(sentence) - length + 1):
            add_sent = ""
            add_choice = True
            #去除标点
            for sent in sentence[idx: idx + length]:
                if sent in punc_dicts :
                    add_choice = False
                    continue
                else:
                    add_sent = add_sent +" "+sent
            if add_choice:
                rst.append(add_sent[1:])
    return rst

def delete_stop(sent):
    stop_words = set(stopwords.words('english'))
    fil_sent = [w for w in sent if w not in stop_words]
    return fil_sent

def delete_punc(sent):
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
    punc_dicts = [i for i in punctuation]
    fil_sent = [w for w in sent if w not in punc_dicts]
    return fil_sent

def ques_text_extract(ques_text):
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
    punc_dicts = [i for i in punctuation]
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(ques_text)
    ques_one_concept = []
    ques_try_concept = []

    for w in words:
        if w.lower() not in stopWords:
            ques_try_concept.append(w.lower())

    #获取 单个词，去除stopword和标点
    for w in words:
        if w.lower() not in stopWords and w.lower() not in punc_dicts:
            ques_one_concept.append(w.lower())
    #获取n-garm分词
    ques_all_concept = cut_by_ngram(ques_try_concept, 2, 4)
    ques_concept = ques_one_concept + ques_all_concept
    #原句
    ques_concept.append(ques_text)

    return ques_concept


def ques_cell(ques_path,ques_cell_path):
    dic = dict()
    with open(ques_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    for idx,ques in enumerate(questions) :
        ques_text = ques["question"]
        q_cell = ques_text_extract(ques_text)
        dic[idx] = {
            "keywords": q_cell,
            'data_item': ques
        }
    with open(ques_cell_path, 'w', encoding='utf-8') as fw:
        json.dump(dic, fw, indent=4, ensure_ascii=False)


def extract_title(all_path,ent_path):
    '''
    build the set of titles
    '''
    with open(all_path, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    set_title=[]
    logger.info("start split and extract titles")
    for table in tqdm.tqdm(tables.values()):
        ori_tit = table["title"].lower().strip()
        set_title.append(ori_tit)
        #是否加入去掉数字的
        # de_tit = delete_num(ori_tit)
        # if de_tit != ori_tit:
        #     set_title.append(de_tit)
    #去重
    set_title = list(set(set_title))
    print("实体数量为{}".format(len(set_title)))
    with open(ent_path, 'w', encoding='utf-8') as fw:
        for tit in set_title:
            fw.write(tit)
            fw.write("\n")

def delete_num(string):
    year_re_str = '\d{4}[–/]\d{2,4}'
    year_re_str1 = '\d{4}[-/]\d{2,4}'
    huge_re_str = '\d{1,3}[,]\d{1,3}'
    time_re_str = '\d{2}[:]\d{2}'
    pattern_re_str = '\d{4}'
    num = re.findall(pattern_re_str, string)
    year = re.findall(year_re_str,string)
    year1 = re.findall(year_re_str1, string)
    huge = re.findall(huge_re_str,string)
    time = re.findall(time_re_str,string)
    all_num = year +year1 + huge + time + num
    if len(all_num) != 0:
        for n in all_num:
            string2 = string.strip(n).strip()
            string = string2
    else:
        string2 = string
    string2 = string2.replace('list of', '').strip()
    return string2

def extract_nq_title(all_path,ent_path):
    '''
    build the set of titles
    '''
    logger.info("start split and extract titles")
    set_title=[]
    with open(all_path, 'r', encoding='utf-8') as f:
        for line in f:
            table = json.loads(line)
            ori_tit = table["documentTitle"].lower().strip()
            set_title.append(ori_tit)
            #是否加入去掉数字的
            de_tit = delete_num(ori_tit)
            if de_tit != ori_tit:
                set_title.append(de_tit)
    #去重
    print("实体数量为{}".format(len(set_title)))
    set_title = list(set(set_title))
    print("实体数量为{}".format(len(set_title)))
    with open(ent_path, 'w', encoding='utf-8') as fw:
        for tit in set_title:
            fw.write(tit)
            fw.write("\n")


if __name__ == '__main__':
    #extract_dev("./data/OTT-QA/all_plain_tables.json","./data/OTT-QA/train_dev_test_table_ids.json","./data/OTT-QA/dev_plain_tables.json")
    #extract_traindev("./data/OTT-QA/all_plain_tables.json","./data/OTT-QA/train_dev_test_table_ids.json","./data/OTT-QA/traindev_plain_tables.json")
    #extract_ent("./data/OTT-QA/traindev_plain_tables.json","./data/OTT-QA/traindev_concept.txt")

    # #抽取表格中的实体
    # extract_ent("./data/OTT-QA/all_plain_tables.json", "./data/OTT-QA/all_concept.txt")

    #抽取标题
    #extract_title("./data/OTT-QA/all_plain_tables.json", "./data/OTT-QA/all_title1.txt")
    #extract_nq_title("./data/NQ/tables/tables.jsonl","./data/NQ/nq_title.txt")

    # #抽取问题中的实体
    ques_cell("./data/OTT-QA/ques/dev_1000.json","./data/OTT-QA/ques/n_gram_keyword_ottqa.json")
