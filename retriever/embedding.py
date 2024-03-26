from tool import SimCSE
import time
#import chromadb
import faiss
import json
import copy
from tqdm import tqdm
import re
model = SimCSE("../sup-simcse-bert-base-uncased/")

#all_concept完成向量库大约需要4个小时
def safe_index(sentences,safe_dir = True,index_file: str = None):
    start = time.time()
    #gpu版本会导致存储index文件不成功。
    if safe_dir :
        index = model.build_index(sentences, use_faiss=True,faiss_fast= False,safe_index=True,index_path= index_file)
    else:
        index = model.build_index(sentences, use_faiss=True,faiss_fast= False,safe_index=False)
    end = time.time()
    print("保存向量库的总时间为{}".format(end-start))
    return index

def word_two(data,numdata):
    l = len(data)
    res = [data[i] + " " + data[j] for i in range(l) for j in range(i + 1, l)]
    res1 = [x+" "+y for x in numdata for y in data]
    return res +res1

def read_entfile(ent_path):
    set_ent = []
    with open(ent_path, 'r', encoding='utf-8') as f:
        ents = f.readlines()
        for ent in ents:
            ent = ent.strip('\n')
            set_ent.append(ent)
    return set_ent

def trans_keyword(keyword_path):
    with open(keyword_path, 'r', encoding='utf-8') as f:
        dics = json.load(f)

    generation_dict = dict()
    for eid, dic in dics.items():
        set_words = dic['generations'][0].split(", ")
        generation_dict[eid] = {
            'keywords': set_words,
            'data_item': copy.deepcopy(dic["data_item"])
        }
    with open(keyword_path, 'w', encoding='utf-8') as f:
        json.dump(generation_dict, f, indent=4)
    print("文件转化结束")

def match_test():
    all_concept = read_entfile("nq_title.txt")
    print(len(all_concept))
    #

    # print(index)
    # len_index = len(index["sentences"])
    # lst = list(range(1,len_index+1))
    # index_ids= [str(j) for j in lst]
    # index_embedding = index["index"].tolist()

    # 读取faiss本地文件
    reading_start = time.time()
    # index1 = faiss.read_index("./all_concept.index")  # 读入index_file.index文件
    # print(index1)
    #成功运行，可以读取本地储存的向量库了。
    index = model.read_index(all_concept,index_file="./nq_title.index")
    reading_end = time.time()
    print("读取向量库的总时间为{}".format(reading_end - reading_start))

    test_ent = ["one tree hill", "babies on board", "scooby doo", "game of thrones","battle of saratoga"]

    start = time.time()
    #根据目标实体列表搜索
    results = model.search(test_ent, top_k=2, threshold=0.6)
    for i, result in enumerate(results):
        print("对于词语: {}".format(test_ent[i]))
        for sentence, score in result:
            print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
        print("")

    end =time.time()
    print("计算相似度的总时间为{}".format(end-start))

def p_result(entity,result):
    set_en = []
    set_se = []
    set_sc = []
    for i, re in enumerate(result):
        ent = entity[i]
        if len(re) != 0 :
            for sent, score in re:
                score = '%.4f' % score
                set_en.append(ent)
                set_se.append(sent)
                set_sc.append(score)
    return set_en,set_se,set_sc


def match_ent(dics,ent_txt,ent_index):
    all_concept = read_entfile(ent_txt)
    #读取本地储存的向量库。
    reading_start = time.time()
    index = model.read_index(all_concept,index_file=ent_index)
    reading_end = time.time()
    print("读取向量库的总时间为{}".format(reading_end - reading_start))

    generation_dict = dict()
    for eid, dic in tqdm(dics.items()):
        ori_ques = dic['data_item']['question'].lower()
        set_oneword = [i.lower() for i in dic['keywords']]
        set_pairword = ex_num(ori_ques)
        ori_ques = [ori_ques]
        set_twoword = word_two(set_oneword,set_pairword)
        set_allword =set_oneword +set_twoword
        if dataset == "OTT-QA":
            all_result = model.search(set_allword,top_k=2,threshold=0.7)
            ori_result = model.search(ori_ques,top_k=5,threshold=0.6)
            all_sent,all_match,all_score = p_result(set_allword,all_result)
            ori_sent, ori_match, ori_score = p_result(ori_ques, ori_result)
            set_words = set_allword + ori_ques
            set_match = all_match + ori_match
            set_score = all_score + ori_score
            # one_result = model.search(set_oneword,top_k=3,threshold=0.7)
            # ori_result = model.search(ori_ques,top_k=5,threshold=0.6)
            # one_sent,one_match,one_score = p_result(set_oneword,one_result)
            # ori_sent, ori_match, ori_score = p_result(ori_ques, ori_result)
            # set_words = set_oneword + ori_ques
            # set_match = one_match + ori_match
            # set_score = one_score + ori_score
        if dataset == "NQ":
            one_result = model.search(set_oneword, top_k=4, threshold=0.6)
            one_sent,one_match,one_score = p_result(set_oneword,one_result)
            if len(set_twoword) > 0 :
                two_result = model.search(set_twoword, top_k=2, threshold=0.7)
                two_sent,two_match,two_score = p_result(set_twoword,two_result)
            else:
                two_sent = two_match = two_score = []
            ori_result = model.search(ori_ques,top_k=2,threshold=0.6)
            ori_sent, ori_match, ori_score = p_result(ori_ques, ori_result)
            set_words = set_allword + ori_ques
            set_match = one_match + two_match + ori_match
            set_score = one_score + two_score + ori_score
        if len(set_match) < 5:
            sup_result = model.search(set_allword,top_k=3,threshold=0.5)
            sup_sent,sup_match,sup_score = p_result(set_allword,sup_result)
            set_match = set_match + sup_match
            set_score = set_score + sup_score

        generation_dict[eid] = {
            'keywords': set_words,
            'matchwords':set_match,
            'score':set_score,
            'data_item': copy.deepcopy(dic["data_item"])
        }
    return generation_dict


def ex_num(string):  #提取年份等特殊字符
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
    all_num = num +year +year1 + huge + time
    all_word = list(set(all_num))
    return all_word

def distin(keyword_path,out_path):
    with open(keyword_path, 'r', encoding='utf-8') as f:
        dics = json.load(f)
    no_dict = dict()
    yes_dict = dict()
    for eid, dic in dics.items():
        ori_ques = dic['data_item']['question'].lower()
        set_pairword = ex_num(ori_ques)
        if len(set_pairword) == 0:
            no_dict[eid]=copy.deepcopy(dic)
        else:
            yes_dict[eid] = copy.deepcopy(dic)

    if dataset == "OTT-QA":
        yes_gen = match_ent(yes_dict,"./all_title1.txt","./all_title1.index")
        no_gen = match_ent(no_dict,"./all_title.txt","./all_title.index")

    if dataset == "NQ":
        yes_gen = match_ent(yes_dict,"./nq_title1.txt","./nq_title1.index")
        no_gen = match_ent(no_dict,"./nq_title.txt","./nq_title.index")

    yes_gen.update(no_gen)
    #恢复顺序
    sort_dict = sorted(yes_gen.items(), key=lambda x: int(x[0]))
    generation_dict = dict((v, k) for v, k in sort_dict)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(generation_dict, f, indent=4,ensure_ascii=False)



if __name__ == "__main__":
    dataset = "OTT-QA"
    if dataset =="OTT-QA":
        #使用知识库中提取的实体列表，利用faiss存储实体向量库,只用运行一次
        # all_concept = read_entfile("./all_title.txt")
        # index = safe_index(all_concept,index_file="./all_title.index")
        # all_concept1 = read_entfile("./all_title1.txt")
        # print(len(all_concept1))
        # index1 = safe_index(all_concept1,index_file="./all_title1.index")

        # similarities = model.similarity(["sevens grand prix series","championship","team challenge cup"], ["What did the 2nd championship win at the Sevens Grand Prix Series for the team with the most top 4 finishes qualify them for ?"])
        # print(similarities)

        #test_path = "./gpt_keyword_ottqa_dev.json"
        test_path = "./gpt_keyword_ottqa_dev.json"

        #转变openai后keyword文件的格式，只用运行一次
        #trans_keyword(test_path)

        out_path = "./faiss_one_ottqa_dev.json"

        distin(test_path,out_path)
    if dataset =="NQ":
        # #使用知识库中提取的实体列表，利用faiss存储实体向量库,只用运行一次
        # all_concept = read_entfile("./nq_title.txt")
        # index = safe_index(all_concept,index_file="./nq_title.index")
        # all_concept1 = read_entfile("./nq_title1.txt")
        # print(len(all_concept1))
        # index1 = safe_index(all_concept1,index_file="./nq_title1.index")

        test_path = "./gpt_keyword_nqtables_test.json"

        # # 转变openai后keyword文件的格式，只用运行一次
        # trans_keyword(test_path)

        out_path = "./faiss_keyword_nqtables_test.json"

        distin(test_path, out_path)

    #以下为测试
    #match_test()

    # ent = ["aaa","bbb"]
    # res = [[["abc",0.9],["acd",0.8]],[]]
    # en,se,sc = p_result(ent,res)
    # print(en)
    # print(se)
    # print(sc)


