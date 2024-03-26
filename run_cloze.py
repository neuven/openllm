import argparse
import os
import time
import json
import copy
from typing import List, Dict
import multiprocessing

ROOT_DIR = os.path.join(os.path.dirname(__file__))
from generation.generator import Generator
from generation.utils import dict2df,table_numrow,num_tokens_from_string

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

            #提取前几行
            subtable_text = data_item['table']
            numrow_subtable = table_numrow(subtable_text,args.num_rows)

            prompt ="/*\ntable caption :" + table_name + "\n" + numrow_subtable + "*/\n" + "Q:" + question +"\nsub questions :"

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



    # Load openai keys
    keys = args.api_keys
     # Annotate
    generator = Generator(args, keys=keys)
    # Map data to different processing   将数据分为default=20的不同进程
    # dataset = random.sample(dataset,100)
    generate_eids = list(range(len(dataset)))
    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[int(g_eid) % args.n_processes].append(g_eid)

    print('\n******* Annotating *******')
    print(len(dataset))
    g_dict = dict()
    worker_results = []

    pool = multiprocessing.Pool(processes=args.n_processes)
    for pid in range(args.n_processes):
        #这里是用来算token的，可用openai自带的来算
        #from transformers import AutoTokenizer
        #tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='../../utils_file/gpt2')
        worker_results.append(pool.apply_async(worker, args=(
            pid,
            args,
            generator,
            generate_eids_group[pid],
            dataset,
        )))

        # Merge annotation results
    for r in worker_results:
        worker_g_dict = r.get()
        g_dict.update(worker_g_dict)
    pool.close()
    pool.join()

    # Save annotation results得到答案文件。
    save_file_name = f'gpt_{args.select_type}_{args.dataset}_{args.dataset_split}.json'
    with open(os.path.join(args.save_dir,save_file_name), 'w') as f:
        json.dump(g_dict, f, indent=4)

    print(f"Elapsed time: {time.time() - start_time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # File path or name文件位置
    parser.add_argument('--dataset', type=str, default='wikitq',
                    choices=['wikitq', 'tab_fact'])
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_keys', type=str, default='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')
    parser.add_argument('--prompt_file', type=str, default='template/cloze.txt')
    parser.add_argument('--save_dir', type=str, default='cloze/')

    # Multiprocess options、
    #进程数，这个地方是工作。用pool eid参数来做，可以达到的效果是同一时间，去处理不同的数据。这周就做这个。
    parser.add_argument('--n_processes', type=int, default=1)


    # Prompt options
    parser.add_argument('--select_type', type=str, default='cloze',
                        choices=['col', 'row', 'all','cloze'])
    #########################
    #######################
    parser.add_argument('--num_rows', type=int, default=3)
    parser.add_argument('--n_shots', type=int, default=14)
    parser.add_argument('--seed', type=int, default=42)

    # gpt options
    parser.add_argument('--engine', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--n_parallel_prompts', type=int, default=2)

    parser.add_argument('--max_generation_tokens', type=int, default=80)
    parser.add_argument('--max_api_total_tokens', type=int, default=4000)
    parser.add_argument('--temperature', type=float, default=0.4)
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


