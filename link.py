import argparse
import json
import re
import copy
import numpy as np
from tqdm import tqdm
from BM25 import BM25
from preprocess import split_into_sentence,delete_stop,delete_punc
from generation.utils import num_tokens_from_string

def find_passage(table_id,table,passages,q_cell):
    g_dict = dict()
    e_dict = dict()
    sg_dict = dict()
    se_dict = dict()
    for row_idx,data in enumerate(table["data"]):
        for col_idx,en in enumerate(data):
            ent = en[0]
            wiki_id = en[1]
            en_len = delete_punc(ent.strip().split(" "))
            if len(en_len) > 10:
                ent= 'row {}, '.format(row_idx + 1) +'col {}'.format(col_idx + 1)
                #是否直接忽略？
                continue
            if len(wiki_id) > 0:
                wiki_passage = ""
                swiki_passage = ""
                for w_id in wiki_id:
                    #去掉/wiki/
                    w_name = w_id[6:]
                    #每个wiki连接取两个句子
                    w_p = passages[w_id]
                    #句子太长时，使用BM25减少句子
                    if num_tokens_from_string(w_p) > 300:
                        p_set = split_into_sentence(w_p)
                        sw_p = passage_BM25(q_cell,p_set)
                    else:
                        sw_p = w_p
                    sw_passage = w_name + ", " + sw_p + "\n"
                    w_passage = w_name +", " +w_p+"\n"
                    wiki_passage += w_passage
                    swiki_passage += sw_passage
                e_dict[ent]=wiki_passage
                if num_tokens_from_string(swiki_passage) > 500:
                    small_p = truncateSentence(swiki_passage, 150)
                else:
                    small_p = swiki_passage
                se_dict[ent] = small_p
    g_dict[table_id] = e_dict
    sg_dict[table_id] = se_dict
    return e_dict,se_dict

def passage_BM25(ques_cell,passage):
    #示例
    # ques_cell = ["created","series","character","robert","played","actor","nonso","anozie","appeared","name"]
    # passage = ["text3","Prime Suspect is a British police procedural television drama series devised by Lynda La Plante.","text2"]
    # search_passage = [["stars","Helen","one","first"],
    #                   ["Prime","Suspect","series","British","police","procedural",'television', "devised"],
    #                   ["stars", "Helen", "one", "first"], ['Occupation', "BAFTA", "three-part", "drama"]]
    search_passage = []
    for pa in passage:
        s_pa = pa.lower().strip().split(' ')
        search_passage.append(s_pa)

    bm = BM25(search_passage)
    score = bm.score_all(ques_cell)
    arr = np.array(score)
    #取分数不为0的句子
    index = list(np.argsort(-arr))
    for idx,i in enumerate(arr):
        if i == 0:
            index.remove(idx)


    if "name" in ques_cell:
        s_passage = passage[0]
        if 0 in index:
            #index = np.delete(index, np.where(index == 0), axis=0)
            index.remove(0)
    else:
        s_passage =""

    #如果分数全为0或者没有选择，只取第一句话
    if np.isin(arr, [0]).all() or len(index) == 0:
        s_passage = passage[0]
    else:
        for i in index:
            s_pass = passage[i]
            s_passage += "."+s_pass

    return s_passage.strip()


def table_linearization(table):
    """
    linearization table according to format.
    """
    linear_table = []
    title = 'title : ' + str(table["title"])
    linear_table.append(title)
    columns = [str(i[0]) for i in table["header"]]
    header = 'col : ' + ' | '.join(columns)
    linear_table.append(header)
    rows = table["data"]
    for row_idx, row in enumerate(rows):
        strrow = [str(j[0]) for j in row]
        #替换太长的实体
        for si,str_ent in enumerate(strrow):
            en_len = delete_punc(str_ent.strip().split(" "))
            if len(en_len)> 20:
                strrow[si] = "#"
        line = 'row {} : '.format(row_idx + 1) + ' | '.join(strrow)
        # if row_idx != len(rows) - 1:
        #     line += '\n'
        linear_table.append(line)

    return linear_table


def find_hyperlink(rerank_path,link_path,passage_path,out_path,sout_path):
    with open(rerank_path, 'r', encoding='utf-8') as f1:
        questions = json.load(f1)
    with open(link_path, 'r', encoding='utf-8') as f2:
        tables = json.load(f2)
    td_tables = [td for td in tables]
    with open(passage_path, 'r', encoding='utf-8') as f3:
        wikis = json.load(f3)
    generation_dict = dict()
    sgeneration_dict = dict()
    for id in tqdm(questions):
        if len(questions[id]["search_id"]) > 0:
            ori_q = questions[id]["data_item"]["question"].lower().strip().split(' ')
            ori_q_cell1 = delete_stop(ori_q)
            ori_q_cell = delete_punc(ori_q_cell1)
            t_dict = dict()
            st_dict = dict()
            topk_id = questions[id]["search_id"][:50]
            ground_id = questions[id]["data_item"]["table_id"]
            if ground_id in topk_id:
                for s_id in topk_id:
                    if s_id in td_tables:
                        table = tables[s_id]
                        table_wiki,stable_wiki = find_passage(s_id,table,wikis,ori_q_cell)
                        table_linear = table_linearization(table)
                        t_dict[s_id] = {
                            "table": table_linear,
                            "passage": table_wiki
                        }
                        st_dict[s_id] = {
                            "table": table_linear,
                            "passage": stable_wiki
                        }
                generation_dict[id] = {
                    "search_tables": t_dict,
                    "acc": "true",
                    "data_item": copy.deepcopy(questions[id]["data_item"])
                }
                sgeneration_dict[id] = {
                    "search_tables": st_dict,
                    "acc": "true",
                    "data_item": copy.deepcopy(questions[id]["data_item"])
                }
            else:
                generation_dict[id] = sgeneration_dict[id] = {
                    "search_tables": [],
                    "acc": "false",
                    "data_item": copy.deepcopy(questions[id]["data_item"])
                }
        else:
            generation_dict[id] = sgeneration_dict[id] ={
                "search_tables": [],
                "acc": "false",
                "data_item": copy.deepcopy(questions[id]["data_item"])
            }
    #全文本
    with open(out_path, 'w', encoding='utf-8') as fw:
        json.dump(generation_dict, fw, indent=4,ensure_ascii=False)
    #筛选文本
    with open(sout_path, 'w', encoding='utf-8') as fw:
        json.dump(sgeneration_dict, fw, indent=4,ensure_ascii=False)

def truncateSentence(s: str, k: int) -> str:
    str_arr = s.split(' ')
    t_str = str_arr[0:k]+str_arr[-k:]
    return ' '.join(t_str)

def chunk_passage(link_table,link_passage):
    link_id = [l_id for l_id in link_passage]
    link_p = ""
    dict_p = dict()
    for en_idx, ent in enumerate(link_table):
        if ent in link_id:
            dict_p[ent] = link_passage[ent]
            l_pa = link_passage[ent]
            #print(num_tokens_from_string(l_pa))
            if num_tokens_from_string(l_pa)>500:
                #print(num_tokens_from_string(l_pa))
                link_p1 = truncateSentence(l_pa,200)
            else:
                link_p1 = l_pa
            #自带\n，不用再加
            link_p +=str(ent) + " : " + link_p1
            #print(num_tokens_from_string(link_p1))
    return link_p, dict_p

def one_table_chunk(tables,t_id,a,title,col,passage,step):
    one_chunk_dict = dict()
    del_step = False
    for i in range(2, len(a), step):
        rows = tables[t_id]["table"][i:i + step]
        ens = []
        for row in rows:
            line_en = re.split(r'[|]',row[8:])
            for en in line_en:
                en1 =en.strip()
                ens.append(en1)
        row_text = "/*\n" + title + "\n" + col + "\n" + "\n".join(rows) + "\n*/"
        #ents = ent_data[i - 2:i + step - 2]
        if passage != [] :
            passage_text, passage_set = chunk_passage(ens, passage)
            prompt = row_text + "\ndescription :\n" + passage_text
        else:
            prompt = row_text + "\n"
        num_token = num_tokens_from_string(prompt)
        if num_token > 1500:
            del_step = True
        # if del_step == True and step ==1 :
        #     print(num_token)
        one_chunk_dict[str(i - 1)] = prompt
    return one_chunk_dict,del_step


def table_chunk(questions,link_path,step):
    '''
    prompt_dict:{
        "table_1":{
                "1":prompt1
                "4":prompt2
                ...
                    }
        "table_2":
        ...
    }
    '''
    # with open(link_path, 'r', encoding='utf-8') as f2:
    #     td_tables = json.load(f2)
    prompt_dict = dict()
    ground_dict = dict()
    tables = questions["search_tables"]
    ground_table = questions["data_item"]["table_id"]
    chunk_len = 0
    if tables != []:
        for t_id in tables:
            if tables[t_id] == {}:
                continue
            title = tables[t_id]["table"][0]
            col = tables[t_id]["table"][1]
            #ent_data = td_tables[t_id]["data"]
            if "passage" in tables[t_id]:
                passage = tables[t_id]["passage"]
            else:
                passage = []
            a = list(range(len(tables[t_id]["table"])))
            #单个表格的分解
            chunk_dict,del_step = one_table_chunk(tables, t_id, a, title, col, passage, step)
            t_step = step
            #如果token太多，再分解行数
            while del_step == True and t_step > 1:
                t_step = t_step-1
                chunk_dict, del_step = one_table_chunk(tables, t_id, a, title, col, passage, t_step)
            chunk_len = chunk_len + len(chunk_dict)
            prompt_dict[t_id]=chunk_dict
            if t_id == ground_table:
                ground_dict[t_id]=chunk_dict
                ground_len = len(chunk_dict)
        #分块太多了
        if chunk_len >80:
            prompt_dict = dict()
            prompt_dict.update(ground_dict)
    return prompt_dict,chunk_len


def nq_linearization(table,q_cell):
    """
    linearization table according to format.
    """
    linear_table = []
    s_linear_table = []
    title = 'title : ' + str(table["documentTitle"])
    linear_table.append(title)
    s_linear_table.append(title)
    columns = [str(i["text"]) for i in table["columns"]]
    for sc, str_col in enumerate(columns):
        if str_col == "" or str_col == " ":
            columns[sc] = "#"
    header = 'col : ' + ' | '.join(columns)
    linear_table.append(header)
    s_linear_table.append(header)
    rows = table["rows"]
    for row_idx, row in enumerate(rows):
        strrow = [str(j["text"]) for j in row["cells"]]
        for si,str_ent in enumerate(strrow):
            if str_ent == "" or str_ent == " ":
                strrow[si] = "#"
        line = 'row {} : '.format(row_idx + 1) + ' | '.join(strrow)
        #筛选过长的实体
        for si,str_ent in enumerate(strrow):
            en_len = delete_punc(str_ent.strip().split(" "))
            if len(en_len)> 100:
                p_set = split_into_sentence(str_ent)
                s_str_ent = passage_BM25(q_cell, p_set)
                strrow[si] = s_str_ent
        s_line = 'row {} : '.format(row_idx + 1) + ' | '.join(strrow)
        # if row_idx != len(rows) - 1:
        #     line += '\n'
        linear_table.append(line)
        s_linear_table.append(s_line)
    return linear_table, s_linear_table

def nq_linear(rerank_path,link_path,out_path,sout_path):
    with open(rerank_path, 'r', encoding='utf-8') as f1:
        questions = json.load(f1)
    table_dict = dict()
    td_tables = []
    with open(link_path, 'r', encoding='utf-8') as f2:
        for line in f2:
            data = json.loads(line)
            t_id = data["table"]["tableId"]
            td_tables.append(t_id)
            table_dict[t_id] = data["table"]

    generation_dict = dict()
    sgeneration_dict = dict()
    for id in tqdm(questions):
        if len(questions[id]["search_id"]) > 0:
            ori_q = questions[id]["data_item"]["question"].lower().strip().split(' ')
            ori_q_cell1 = delete_stop(ori_q)
            ori_q_cell = delete_punc(ori_q_cell1)
            t_dict = dict()
            st_dict = dict()
            topk_id = questions[id]["search_id"][:50]
            ground_id = questions[id]["data_item"]["table_id"]
            if ground_id in topk_id:
                for s_id in topk_id:
                    if s_id in td_tables:
                        table = table_dict[s_id]
                        table_linear, s_table_linear = nq_linearization(table,ori_q_cell)
                        t_dict[s_id] = {
                            "table": table_linear,
                        }
                        st_dict[s_id] = {
                            "table": s_table_linear,
                        }
                    if s_id == ground_id :
                        break
                generation_dict[id] = {
                    "search_tables": t_dict,
                    "acc": "true",
                    "data_item": copy.deepcopy(questions[id]["data_item"])
                }
                sgeneration_dict[id] = {
                    "search_tables": st_dict,
                    "acc": "true",
                    "data_item": copy.deepcopy(questions[id]["data_item"])
                }
            else:
                generation_dict[id] = sgeneration_dict[id] = {
                    "search_tables": [],
                    "acc": "false",
                    "data_item": copy.deepcopy(questions[id]["data_item"])
                }
        else:
            generation_dict[id] = sgeneration_dict[id] ={
                "search_tables": [],
                "acc": "false",
                "data_item": copy.deepcopy(questions[id]["data_item"])
            }
    #全文本
    with open(out_path, 'w', encoding='utf-8') as fw:
        json.dump(generation_dict, fw, indent=4,ensure_ascii=False)
    #筛选文本
    with open(sout_path, 'w', encoding='utf-8') as fw:
        json.dump(sgeneration_dict, fw, indent=4,ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nqtables',
                    choices=['nqtables', 'ottqa'])
    args = parser.parse_args()

    if args.dataset == 'ottqa':
        rerank_path = "./data/OTT-QA/BM25/reranker_ottqa_dev.json"
        link_path = "./data/OTT-QA/traindev_tables.json"
        passage_path = "./data/OTT-QA/all_passages.json"
        table_link_path = "./data/OTT-QA/link/linear_ottqa_dev.json"
        small_table_link_path = "./data/OTT-QA/link/small_linear_ottqa_dev.json"
        #根据黄金连接找文本
        #find_hyperlink(rerank_path,link_path,passage_path,table_link_path,small_table_link_path)

    if args.dataset == 'nqtables':
        rerank_path = "./data/NQ/BM25/reranker_nqtables_test.json"
        link_path = "./data/NQ/interactions/test.jsonl"
        table_link_path = "./data/NQ/link/linear_nqtables_test.json"
        small_table_link_path = "./data/NQ/link/small_linear_nqtables_test.json"
        #表格文本
        #nq_linear(rerank_path,link_path,table_link_path,small_table_link_path)


    #with open('data/OTT-QA/qa/subtable_ottqa_dev2.json', 'r', encoding='utf-8') as f1:
    with open('data/NQ/qa/subtable_nqtables_test.json', 'r', encoding='utf-8') as f1:
        questions = json.load(f1)
    c_len = 0
    for a_id in tqdm(questions):
        # if int(a_id) >600 :
        #     continue
        que = questions[a_id]
        text,step_len = table_chunk(que,link_path,step=15)
        #print(text)
        if step_len >50:
            print(a_id)
            print(step_len)
        c_len = c_len+step_len
    print("总共分块个数为{}".format(c_len))