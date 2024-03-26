import json
import re
import collections
import string
import sys

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_list_exact(a_gold, a_pred) :
    a_pred = normalize_answer(a_pred)
    exact = True
    for go in a_gold:
        go = normalize_answer(go)
        if go not in a_pred:
            exact = False
            break
    return int(exact)

def compute_list_f1(a_gold, a_pred):
    pred_toks = get_tokens(a_pred)
    gold_text = " ".join(a_gold)
    gold_toks = get_tokens(gold_text)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_qafile_scores(reference):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}
    first_exact_scores = {}
    first_f1_scores = {}
    qid_list = []

    for idx in reference:
        qas_id = reference[idx]["data_item"]["question_id"]
        qid_list.append(qas_id)
        gold_answers = reference[idx]["data_item"]["answer-text"]
        prediction = reference[idx]["generations"]

        first_exact_scores[qas_id] = compute_exact(gold_answers, prediction[0])
        first_f1_scores[qas_id] = compute_f1(gold_answers, prediction[0])
        exact_scores[qas_id] = max(compute_exact(a, gold_answers) for a in prediction)
        f1_scores[qas_id] = max(compute_f1(a, gold_answers) for a in prediction)

    total = len(qid_list)

    for k in qid_list:
        if k not in exact_scores:
            print("WARNING: MISSING QUESTION {}".format(k))
    qid_list = list(set(qid_list) & set(exact_scores.keys()))

    return collections.OrderedDict(
        [
            ("first exact", 100.0 * sum(first_exact_scores[k] for k in qid_list) / total),
            ("first f1", 100.0 * sum(first_f1_scores[k] for k in qid_list) / total),
            ("total exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ("total f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ("total", total),
        ]
    )

def get_nq_scores(reference):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}
    oracle_exact_scores = {}
    oracle_f1_scores = {}
    qid_list = []

    for idx in reference:
        qas_id = reference[idx]["data_item"]["question_id"]
        qid_list.append(qas_id)
        gold_answers = [reference[idx]["data_item"]["answer-text"]]
        for alter_gold_answers in reference[idx]["data_item"]["alternativeAnswers"]:
            gold_answers.append(alter_gold_answers)
        prediction = reference[idx]["generations"][0]

        if len(gold_answers[0]) >1 :
            exact_scores[qas_id] = max(compute_list_exact(g,prediction) for g in gold_answers)
            f1_scores[qas_id] = max(compute_list_f1(g, prediction) for g in gold_answers)
        else:
            exact_scores[qas_id] = max(compute_exact(g[0], prediction) for g in gold_answers)
            f1_scores[qas_id] = max(compute_f1(g[0], prediction) for g in gold_answers)

        temp_set_ex = []
        temp_set_f1 = []
        for pred in reference[idx]["generations"]:
            if len(gold_answers[0]) > 1:
                temp_set_ex.append(max(compute_list_exact(g, pred) for g in gold_answers))
                temp_set_f1.append(max(compute_list_f1(g, pred) for g in gold_answers))
            else:
                temp_set_ex.append(max(compute_exact(g[0], pred) for g in gold_answers))
                temp_set_f1.append(max(compute_f1(g[0], pred) for g in gold_answers))
        oracle_exact_scores[qas_id] = max(temp_set_ex)
        oracle_f1_scores[qas_id] = max(temp_set_f1)

    total = len(qid_list)

    for k in qid_list:
        if k not in exact_scores:
            print("WARNING: MISSING QUESTION {}".format(k))
    qid_list = list(set(qid_list) & set(exact_scores.keys()))

    return collections.OrderedDict(
        [
            ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ("oracle exact", 100.0 * sum(oracle_exact_scores[k] for k in qid_list) / total),
            ("oracle f1", 100.0 * sum(oracle_f1_scores[k] for k in qid_list) / total),
            ("total", total),
        ]
    )

# assert len(sys.argv) == 3, "you need to input the file"
#
# with open(sys.argv[1], 'r') as f:
#     data = json.load(f)
#
# with open(sys.argv[2], 'r') as f:
#     ref = json.load(f)
#
# print(get_raw_scores(data, ref))

# with open('data/OTT-QA/qa/result_qa_ottqa_test2.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# print(get_qafile_scores(data))

with open('data/NQ/qa/result_qa_nqtables_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(get_nq_scores(data))



