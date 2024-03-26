from typing import Dict,List
import tiktoken
import pandas as pd

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def table_numrow(text,numrow):
    lines = text.splitlines()
    maxnum = len(lines)
    if maxnum > numrow:
        line2 = lines[:numrow]
    else:
        line2 = lines
    line_text = ''
    for line in line2:
        line_text = line_text +line+'\n'
    return line_text


def dict2df(table: Dict, add_row_id=False, lower_case=True):
    """
    Dict to pd.DataFrame.
    tapex format.
    """
    header, rows = table[0], table[1:]
    
    df = pd.DataFrame(data=rows, columns=header)
    return df

def table_linearization(table: pd.DataFrame, format:str='codex'):
    """
    linearization table according to format.
    """
    assert format in ['tapex', 'codex']
    linear_table = ''
    if format == 'tapex':
        header = 'col : ' + ' | '.join(table.columns) + ' '
        linear_table += header
        rows = table.values.tolist()
        for row_idx,row in enumerate(rows):
            line = 'row {} : '.format(row_idx + 1) + ' | '.join(rows) + ' '
            line += '\n'
            linear_table += line
    elif format == 'codex':
        header = 'col : ' + ' | '.join(table.columns) + '\n'
        linear_table += header
        rows = table.values.tolist()
        for row_idx,row in enumerate(rows):
            line = 'row {} : '.format(row_idx + 1) + ' | '.join(row)
            if row_idx != len(rows) - 1:
                line += '\n'
                
            linear_table += line
    return linear_table

def tapex_post_prepare():
    pass

def twoD_list_transpose(arr:List[List],keep_num_rows:int=3):
    arr = arr[:keep_num_rows+1] if keep_num_rows + 1 <= len(arr) else arr
    return [[arr[i][j] for i in range(len(arr))] for j in range(len(arr[0]))]

def dic2prompt(dic:Dict):
    prompt = ''
    pass