import argparse
import os
import time
import json
import copy
from typing import List, Dict
import multiprocessing
import ast
from collections import Counter

ROOT_DIR = os.path.join(os.path.dirname(__file__))
from generation.generator import Generator
from generation.utils import dict2df,table_numrow,num_tokens_from_string


def max_ans(doc_path):
    with open(doc_path,'r') as f:
        data = json.load(f)

    for key in data:
        generations = data[key]["generations"]
        counts = Counter([gen for gen in generations])  # 将句子转为小写字母并计数
        most_common = counts.most_common(1)  # 找到出现最多的句子
        if most_common:  # 如果有出现最多的句子
            most_common_sentence = most_common[0][0]  # 取出这个句子
        else:  # 如果没有出现最多的句子
            most_common_sentence = generations[-1]  # 取最后一个句子

        data[key]["generations"][0] = most_common_sentence  # 将generations改为只有出现最多的那个句子

    with open(doc_path, 'w') as f:
        json.dump(data, f, indent=4)


def worker(
    pid: int,
    args,
    generator: Generator,
    g_eids: List,
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
            data_item = dataset[g_eid]
            generation_dict[g_eid] = {
                'generations': [],
                'data_item': copy.deepcopy(data_item)
            }
            n_shots = args.n_shots
            few_shot_prompt = generator.build_few_shot_prompt_from_file(
                file_path=args.prompt_file,
                n_shots=n_shots
            )

            question = data_item['question']

            if 'name' in data_item.keys():
                table_name = data_item['name']
            else:
                None

            # #提取前几行
            # subtable_text = data_item['table']
            # numrow_subtable = table_numrow(subtable_text,args.num_rows)

            prompt ="Q:" + question +"\nsub questions :"

            print("生成完整提示")
            user_input = few_shot_prompt + '\n\n' + prompt

            #算token，以下两行的函数不用在意
            max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
            num_token =  num_tokens_from_string(user_input)
            #限制token，减shot
            while num_token >= max_prompt_tokens:  # TODO: Add shrink rows
                n_shots -= 1
                assert n_shots >= 0
                few_shot_prompt = generator.build_few_shot_prompt_from_file(
                    file_path=args.prompt_file,
                    n_shots=n_shots
                )
            user_input = few_shot_prompt + "\n\n" + prompt
            print("*"*80)
            print(prompt)
            built_few_shot_prompts.append((g_eid, user_input))


            print(f"Process#{pid}: Building prompt for eid#{g_eid}, original_id#{data_item['question']}")
            if len(built_few_shot_prompts) < 1:
                continue

            print(f"Process#{pid}: Prompts ready with {len(built_few_shot_prompts)} parallels. Run openai API.")
            response_dict = generator.generate_gpt(
                prompts=built_few_shot_prompts,
                verbose=args.verbose
            )
            generation_dict[g_eid]['generations'] = response_dict

            built_few_shot_prompts = []
        except Exception as e:
            print(f"Process#{pid}: eid#{g_eid}, wtqid#{data_item['question']} generation error: {e}")
    # Final generation inference
    if len(built_few_shot_prompts) > 0:
        response_dict = generator.generate_gpt(
            prompts=built_few_shot_prompts,
            verbose=args.verbose
        )
        generation_dict[g_eid]['generations'] = response_dict

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
    if args.dataset == 'wikitq':
        args.prompt_file = os.path.join(ROOT_DIR, 'data/wikitable',args.prompt_file)
        args.save_dir = os.path.join(ROOT_DIR, 'data/wikitable',args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        if args.dataset_split == 'test':
            with open('./data/wikitable/wikitable_30.json', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    dic = json.loads(line)
                    dataset.append(dic)
    if args.dataset == 'ottqa':
        args.prompt_file = os.path.join(ROOT_DIR, 'data/OTT-QA',args.prompt_file)
        args.save_dir = os.path.join(ROOT_DIR, 'data/OTT-QA',args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        if args.dataset_split == 'test':
            with open('data/OTT-QA/ques/test_50.json', 'r', encoding='utf-8') as f:
                dataset = json.load(f)

    # 从命令行参数中加载OpenAI API密钥
    keys = args.api_keys
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
    with open(os.path.join(args.save_dir,save_file_name), 'w') as f:
        json.dump(dicts, f, indent=4)

    #取次数最多的答案放在第一个
    max_ans(os.path.join(args.save_dir,save_file_name))

    print(f"Elapsed time: {time.time() - start_time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # File path or name文件位置
    parser.add_argument('--dataset', type=str, default='ottqa',
                    choices=['wikitq', 'ottqa'])
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'validation', 'test'])
    #这里存好多个key
    parser.add_argument('--api_keys', type=ast.literal_eval,
                        default=['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',
                                 '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'])
    parser.add_argument('--nucleus_p', type=float, default=0.5, help='nucleus sampling probability (0 to 1)')
    parser.add_argument('--prompt_file', type=str, default='template/ques.txt')
    parser.add_argument('--save_dir', type=str, default='ques/')

    # Multiprocess options、
    #进程数，n个进程可以同时使用不同的n个key

    parser.add_argument('--n_processes', type=int, default=2)


    # Prompt options
    parser.add_argument('--select_type', type=str, default='cloze',
                        choices=['keyword', 'row', 'all','cloze'])
    #########################
    #######################
    parser.add_argument('--num_rows', type=int, default=3)
    #提示个数
    parser.add_argument('--n_shots', type=int, default=14)
    parser.add_argument('--seed', type=int, default=42)

    # CodeX options
    parser.add_argument('--engine', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--n_parallel_prompts', type=int, default=2)

    parser.add_argument('--max_generation_tokens', type=int, default=80)
    parser.add_argument('--max_api_total_tokens', type=int, default=4000)
    parser.add_argument('--temperature', type=float, default=0.4)
    #决定输出多少个可能答案
    parser.add_argument('--sampling_n', type=int, default=20)
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

    main()