import argparse
import os
import re
import time
import json
import copy
from typing import List, Dict
import multiprocessing
import ast
from collections import Counter
from link import table_chunk , chunk_passage

ROOT_DIR = os.path.join(os.path.dirname(__file__))
from generation.generator import Generator
from generation.utils import dict2df,table_numrow,num_tokens_from_string

def is_number(num):
    try:
        int(num)
        str_num = "row "+str(num)
        return str_num
    except ValueError:
        pass
    return num

def sub_table(linear_path,choose_path,sub_path):

    with open(choose_path,'r',encoding='utf-8') as f1:
        choose = json.load(f1)
    with open(linear_path,'r',encoding='utf-8') as f2:
        data = json.load(f2)

    generation_dict = dict()
    for key in choose:
        generations = choose[key]["generations"]
        if len(generations)> 0:
            sub_table_dict = dict()
            for tab in generations:
                table = data[key]["search_tables"][tab]
                gen_tab = generations[tab]
                tab_row = []
                for step in gen_tab:
                    #去中括号
                    row_id = step[1:-1]
                    if row_id:
                        row_id = row_id.split(",")
                        for row_i in row_id :
                            row_i = row_i.strip()
                            if row_i != "*":
                                row_stri = is_number(row_i)
                                tab_row.append(row_stri)
                if len(tab_row) > 0:
                    sub_data = extract_subtab(table, tab_row)
                    sub_table_dict[tab] = sub_data
                else:
                    sub_table_dict[tab] = {}
            generation_dict[key] = {
                "search_tables": sub_table_dict,
                "data_item": copy.deepcopy(data[key]["data_item"])
            }
        else :
            generation_dict[key] = {
                "search_tables": [],
                "data_item": copy.deepcopy(data[key]["data_item"])
            }
    with open(sub_path, 'w',encoding='utf-8') as fw:
        json.dump(generation_dict, fw, indent=4,ensure_ascii=False)


def extract_subtab(c_table, id):
    '''
    sub_data:{
            "table":[...]
            "passage":{...}
    }
    '''
    table_text  = c_table["table"]
    passage_text = c_table["passage"]
    sub_tab = table_text[:2]
    ens = []
    sub_data = dict()
    for t in table_text[2:]:
        t_rowid = t[:6].strip()
        if t_rowid in id :
            sub_tab.append(t)
            line_en = re.split(r'[|]', t[8:])
            for en in line_en:
                en1 = en.strip()
                ens.append(en1)
    if len(ens) > 0 :
        text_subpa, sub_pa = chunk_passage(ens, passage_text)
    else :
        sub_pa = {}

    sub_data["table"] = sub_tab
    sub_data["passage"] = sub_pa

    return sub_data






def max_ans(doc_path, out_path):
    with open(doc_path,'r',encoding='utf-8') as f:
        data = json.load(f)

    p1 = re.compile(r'[(](.*?)[)]', re.S)
    for key in data:
        generations = data[key]["generations"]
        for tab in generations:
            gen_tab = generations[tab]
            ans_tab = []
            step_num = [int(step) for step in gen_tab]
            if len(step_num) > 1:
                step_diff = step_num[1] - step_num[0]
            else:
                step_diff =15
            for step in gen_tab:
                gen_step = gen_tab[step]
                step_ans = []
                for gen in gen_step:
                    string = gen.split("\n")[-1]
                    if string:
                        if string[-1] != ")":
                            print(key)
                    answer = re.findall(p1, string)
                    if answer:
                        answer = answer[0]
                    else:
                        answer = "[*]"
                    step_ans.append(answer)
                step_ans = list(set(step_ans))
                for an in step_ans:
                    if an == "[all]":
                        s_num = [str(ss) for ss in range(int(step), int(step)+step_diff)]
                        an = "[" + ",".join(s_num) +"]"
                    if an != "[*]":
                        ans_tab.append(an)
                # counts = Counter(step_ans)
                # most_common = counts.most_common(1)
                # if most_common:
                #     ans_tab.append(most_common[0][0])
                # else:
                #     ans_tab.append(step_ans[-1])
            data[key]["generations"][tab] = ans_tab

    with open(out_path, 'w',encoding='utf-8') as fw:
        json.dump(data, fw, indent=4, ensure_ascii=False)
    print("-----------choose文件转化完成-----------------")

def worker(
    pid: int,
    args,
    generator: Generator,
    g_eids: List,
    spa_keys : List,
    dataset: List[Dict],
):
    """
    A worker process for annotating.
    """
    #验证1下确实跑了2进程
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print(f"Worker {pid} (pid={os.getpid()}) is processing")
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    generation_dict = dict()
    built_few_shot_prompts = []
    for g_eid in g_eids:
        try:
            data_item = dataset[str(g_eid)]
            n_shots = args.n_shots
            few_shot_prompt = generator.build_few_shot_prompt_from_file(
                file_path=args.prompt_file,
                n_shots=n_shots
            )

            #问题
            ori_question = data_item["data_item"]["question"]
            search_tab = data_item["search_tables"]
            if len(search_tab) == 0:
                generation_dict[g_eid] = {
                    'generations': [],
                    'data_item': copy.deepcopy(data_item["data_item"])
                }
                continue

            #返回了一个dict
            set_prompt, chunk_len= table_chunk(data_item,args.link_path,step=args.d_step)
            table_dict = dict()
            for table_id,table_prompt in set_prompt.items():
                step_dict = dict()
                for step_id,step_prompt in table_prompt.items():

                    prompt = step_prompt+"statement : " + ori_question +"\nexplain : "

                    n1 = num_tokens_from_string(prompt)
                    n2 = num_tokens_from_string(few_shot_prompt)

                    print("生成完整提示")
                    user_input = few_shot_prompt + '\n\n' + prompt

                    #算token，以下两行的函数不用在意
                    max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens*args.sampling_n
                    num_token =  num_tokens_from_string(user_input)
                    #限制token，减shot
                    t_nshot = n_shots
                    while num_token >= max_prompt_tokens and t_nshot > 1:  # TODO: Add shrink rows
                        t_nshot -= 1
                        few_shot_prompt = generator.build_few_shot_prompt_from_file(
                            file_path=args.prompt_file,
                            n_shots=t_nshot
                        )
                        user_input = few_shot_prompt + "\n\n" + prompt
                        num_token = num_tokens_from_string(user_input)
                    if num_token >= max_prompt_tokens:
                        step_dict[step_id] = []
                        continue
                    print("*"*80)
                    print(prompt)
                    built_few_shot_prompts.append((g_eid, user_input))


                    print(f"Process#{pid}: Building prompt for eid#{g_eid}, table_id#{table_id},step_id#{step_id}")
                    if len(built_few_shot_prompts) < 1:
                        step_dict[step_id] = []
                        continue

                    #
                    print(f"Process#{pid}: Prompts ready with {len(built_few_shot_prompts)} parallels. Run openai API.")
                    response_dict = generator.generate_gpt(
                        prompts=built_few_shot_prompts,
                        verbose=args.verbose,
                        spare_keys=spa_keys,
                        time_sleep=20
                    )
                    step_dict[step_id] = response_dict

                    built_few_shot_prompts = []
                table_dict[table_id]=step_dict

            generation_dict[g_eid] = {
                'generations': table_dict,
                'data_item': copy.deepcopy(data_item["data_item"])
            }
            #临时追加
            with open('data/OTT-QA/link/gpt_choose_ottqa_dev.json', 'a', encoding='utf-8') as fw:
                json.dump(generation_dict[g_eid], fw, indent=4, ensure_ascii=False)
                fw.write(",\n")
        finally:
            print("yes")
        # except Exception as e:
        #     print(f"Process#{pid}: eid#{g_eid}, generation error: {e}")

    return generation_dict


def main():
    def twoD_list_transpose(arr):
        return [[arr[i][j] for i in range(len(arr))] for j in range(len(arr[0]))]
    def filter_col(table,pred_col):
        table = twoD_list_transpose(table)
        new_table = []
        for cols in table:
            if cols[0] in pred_col:
                new_table.append(copy.deepcopy(cols))
        if len(new_table) == 0:
            new_table = table
        new_table = twoD_list_transpose(new_table)
        return new_table

    # Load dataset

    dataset = []

    start_time = time.time()
    if args.dataset == 'ottqa':
        args.prompt_file = os.path.join(ROOT_DIR, 'data/OTT-QA',args.prompt_file)
        args.save_dir = os.path.join(ROOT_DIR, 'data/OTT-QA',args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        if args.dataset_split == 'test':
            with open('data/OTT-QA/qa/subtable_ottqa_test.json', 'r', encoding='utf-8') as f:
            #with open('data/OTT-QA/link/test_5.json', 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        if args.dataset_split == 'dev':
            with open('data/OTT-QA/qa/subtable_ottqa_dev.json', 'r', encoding='utf-8') as f:
                dataset = json.load(f)

    # 从命令行参数中加载OpenAI API密钥
    # Load openai keys
    with open(args.api_keys, 'r') as f:
        keys = [line.strip() for line in f.readlines()]
    #备用keys
    with open(args.spare_keys, 'r') as f:
        spare_keys = [line.strip() for line in f.readlines()]
    #keys = args.api_keys
    key_groups = [[key] for key in keys]
    #key_groups = [keys[i:i + 2] for i in range(0, len(keys), 2)]
    # 创建OpenAI生成器对象
    generators = [Generator(args, keys=key_group) for key_group in key_groups]
    #generator = Generator(args, keys=keys)

    # 将数据分为不同组
    generate_eids = list(range(len(dataset)))
    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[int(g_eid) % args.n_processes].append(g_eid)

    # 打印待注释数据的条数
    print('\n******* 正在注释 *******')
    print(len(dataset))

    # 初始化一个空字典，用于存储注释结果
    g_dict = dict()

    # 初始化一个空列表，用于存储每个进程的结果
    worker_results = []

    # 创建一个进程池
    pool = multiprocessing.Pool(processes=args.n_processes)
    for pid in range(args.n_processes):
        # 为每个进程分配一组数据和生成器
        worker_results.append(pool.apply_async(worker, args=(
            pid,
            args,
            generators[pid % len(generators)],  # 使用不同的生成器对象
            #generator,
            generate_eids_group[pid],
            spare_keys,
            dataset,
        )))

    # 将所有 worker 进程的注释结果添加到 `g_dict` 字典中
    for r in worker_results:
        worker_g_dict = r.get()
        g_dict.update(worker_g_dict)

    #恢复顺序
    sort_dict = sorted(g_dict.items(), key=lambda x: int(x[0]))
    dicts = dict((v, k) for v, k in sort_dict)

    # 关闭进程池
    pool.close()
    pool.join()

    # Save annotation results得到答案文件。
    save_file_name = f'gpt_{args.select_type}_{args.dataset}_{args.dataset_split}.json'
    with open(os.path.join(args.save_dir,save_file_name), 'w',encoding='utf-8') as f:
        json.dump(dicts, f, indent=4,ensure_ascii=False)

    #取次数最多的答案放在第一个
    #max_ans(os.path.join(args.save_dir,save_file_name))

    print(f"Elapsed time: {time.time() - start_time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # File path or name文件位置
    parser.add_argument('--dataset', type=str, default='ottqa',
                    choices=['wikitq', 'ottqa'])
    parser.add_argument('--dataset_split', type=str, default='dev', choices=['train', 'dev', 'test'])
    #这里存好多个key
    parser.add_argument('--api_keys', type=str, default='./key_50.txt')
    parser.add_argument('--spare_keys', type=str, default='./key_5.txt')
    parser.add_argument('--nucleus_p', type=float, default=0.5, help='nucleus sampling probability (0 to 1)')
    parser.add_argument('--prompt_file', type=str, default='template/choose.txt')
    parser.add_argument('--save_dir', type=str, default='link/')
    parser.add_argument('--link_path', type=str, default='./data/OTT-QA/traindev_tables.json')

    # Multiprocess options、
    #进程数，n个进程可以同时使用不同的n个key
    parser.add_argument('--n_processes', type=int, default=50)


    # Prompt options
    parser.add_argument('--select_type', type=str, default='choose',
                        choices=['keyword', 'choose', 'qa','cloze'])
    #########################
    #######################
    parser.add_argument('--num_rows', type=int, default=3)
    #提示个数
    parser.add_argument('--n_shots', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    #分解表格行数
    parser.add_argument('--d_step', type=int, default=15)

    # CodeX options
    parser.add_argument('--engine', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--n_parallel_prompts', type=int, default=2)

    parser.add_argument('--max_generation_tokens', type=int, default=150)
    parser.add_argument('--max_api_total_tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.4)
    #决定输出多少个可能答案
    parser.add_argument('--sampling_n', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--stop_tokens', type=str, default='\n\n',
                        help='Split stop tokens by ||')

    # debug options
    parser.add_argument('-v', '--verbose', action='store_false')

    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split('||')


    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))


    #main()
    if args.dataset_split == 'test':
        max_ans("data/OTT-QA/link/gpt_choose_ottqa_test.json","data/OTT-QA/link/result_choose_ottqa_test.json")
        # max_ans("data/OTT-QA/link/gpt_choose_ottqa_test.json","data/OTT-QA/link/result_5.json")
        # 生成分解后的子表
        # sub_table('data/OTT-QA/qa/subtable_ottqa_test.json','data/OTT-QA/link/result_choose_ottqa_test.json','data/OTT-QA/qa/subtable_ottqa_test2.json')
    if args.dataset_split == 'dev':
        #不要随便用，缺了几个
        #max_ans("data/OTT-QA/link/gpt_choose_ottqa_dev2.json", "data/OTT-QA/link/result_choose_ottqa_dev2.json")
        #生成分解后的子表
        #sub_table('data/OTT-QA/link/small_linear_ottqa_dev.json','data/OTT-QA/link/result_choose_ottqa_dev2.json','data/OTT-QA/qa/subtable_ottqa_dev2.json')
        max_ans("data/NQ/link/gpt_choose_nqtables_test.json", "data/NQ/link/result_choose_nqtables_test.json")
