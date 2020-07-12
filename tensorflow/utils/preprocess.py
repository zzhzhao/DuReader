###############################################################################
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module finds the most related paragraph of each document according to recall.
"""
import csv
import sys

import jieba
import re

if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import json
from collections import Counter


def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def recall(prediction, ground_truth):
    """
    This function calculates and returns the recall
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of recall
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[1]


def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of f1
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[2]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        metric_fn: metric function pointer which calculates scores according to corresponding logic.
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def find_best_question_match(doc, question, with_score=False):
    """
    For each document, find the paragraph that matches best to the question.
    Args:
        doc: The document object.
        question: The question tokens.
        with_score: If True then the match score will be returned,
            otherwise False.
    Returns:
        The index of the best match paragraph, if with_score=False,
        otherwise returns a tuple of the index of the best match paragraph
        and the match score of that paragraph.
    """
    most_related_para = -1
    max_related_score = 0
    most_related_para_len = 0
    for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
        if len(question) > 0:
            related_score = metric_max_over_ground_truths(recall,
                    para_tokens,
                    question)
        else:
            related_score = 0

        if related_score > max_related_score \
                or (related_score == max_related_score \
                and len(para_tokens) < most_related_para_len):
            most_related_para = p_idx
            max_related_score = related_score
            most_related_para_len = len(para_tokens)
    if most_related_para == -1:
        most_related_para = 0
    if with_score:
        return most_related_para, max_related_score
    return most_related_para


def find_fake_answer(sample):
    """
    For each document, finds the most related paragraph based on recall,
    then finds a span that maximize the f1_score compared with the gold answers
    and uses this span as a fake answer span
    Args:
        sample: a sample in the dataset
    Returns:
        None
    Raises:
        None
    """
    for doc in sample['documents']:
        most_related_para = -1
        most_related_para_len = 999999
        max_related_score = 0
        for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
            if len(sample['segmented_answers']) > 0:
                related_score = metric_max_over_ground_truths(recall,
                                                              para_tokens,
                                                              sample['segmented_answers'])
            else:
                continue
            if related_score > max_related_score \
                    or (related_score == max_related_score
                        and len(para_tokens) < most_related_para_len):
                most_related_para = p_idx
                most_related_para_len = len(para_tokens)
                max_related_score = related_score
        doc['most_related_para'] = most_related_para

    sample['answer_docs'] = []
    sample['answer_spans'] = []
    sample['fake_answers'] = []
    sample['match_scores'] = []

    best_match_score = 0
    best_match_d_idx, best_match_span = -1, [-1, -1]
    best_fake_answer = None
    answer_tokens = set()
    for segmented_answer in sample['segmented_answers']:
        answer_tokens = answer_tokens | set([token for token in segmented_answer])
    for d_idx, doc in enumerate(sample['documents']):
        if not doc['is_selected']:
            continue
        if doc['most_related_para'] == -1:
            doc['most_related_para'] = 0
        most_related_para_tokens = doc['segmented_paragraphs'][doc['most_related_para']][:1000]
        for start_tidx in range(len(most_related_para_tokens)):
            if most_related_para_tokens[start_tidx] not in answer_tokens:
                continue
            for end_tidx in range(len(most_related_para_tokens) - 1, start_tidx - 1, -1):
                span_tokens = most_related_para_tokens[start_tidx: end_tidx + 1]
                if len(sample['segmented_answers']) > 0:
                    match_score = metric_max_over_ground_truths(f1_score, span_tokens,
                                                                sample['segmented_answers'])
                else:
                    match_score = 0
                if match_score == 0:
                    break
                if match_score > best_match_score:
                    best_match_d_idx = d_idx
                    best_match_span = [start_tidx, end_tidx]
                    best_match_score = match_score
                    best_fake_answer = ''.join(span_tokens)
    if best_match_score > 0:
        sample['answer_docs'].append(best_match_d_idx)
        sample['answer_spans'].append(best_match_span)
        sample['fake_answers'].append(best_fake_answer)
        sample['match_scores'].append(best_match_score)

#加一个将莱斯杯数据转换为dureader格式，即 莱斯杯格式->dureader的raw格式
#注意：测试集和训练集本不同（有无答案），但是我的测试集是从训练集合中分出来的
#故此，可以用同一个函数，如果日后需要将没有答案的测试集进行转换，则需要重写一个函数
def transform_dureader_raw(laisi_path, dureader_path):
    i=0
    with open(laisi_path,'r', encoding='utf-8') as f:
        with open(dureader_path,'w',encoding='utf-8') as w:

            reader = csv.reader(f)
            next(reader, None)
            for line in reader:

                i+=1
                # if i>100:break



                pattern = re.compile('content\d@(.*?)@content')
                answers = re.findall(pattern, line[0])

                documents = []
                for doc, title in zip(line[2:7], line[10:15]):
                    bool_value = False
                    for answer in answers:
                        if doc.find(answer)!=-1 or doc.find(answer[:-1])!=-1:
                            bool_value = True
                    #注意：找不到答案的即五个document的bool_value都是False，如果不做处理，模型也会把他们当成一部分训练数据
                    #另外，find_fake_answer中is_selected=False的不会去找虚假答案，因此额外的处理是
                    #最后看documents里面的is_selected都是False的话，就全设置为True，让其自己找一个虚假答案
                    paragraphs = [para+'。' for para in doc.split('。') if para]

                    documents.append({
                        'is_selected': bool_value,
                        'title':title,
                        'paragraphs':paragraphs,
                        'segmented_title': list(jieba.cut(title)),
                        'segmented_paragraphs': [list(jieba.cut(para)) for para in paragraphs],
                    })
                # 注意：找不到答案的即五个document的bool_value都是False，如果不做处理，模型也会把他们当成一部分训练数据
                # 另外，find_fake_answer中is_selected=False的不会去找虚假答案，因此额外的处理是
                # 最后看documents里面的is_selected都是False的话，就全设置为True，让其自己找一个虚假答案
                sum_is_selected = 0
                for doc in documents:
                    sum_is_selected += doc['is_selected']
                if sum_is_selected == 0:
                    for doc in documents:
                        doc['is_selected'] = True




                sample = {
                    'documents':documents,
                    "question": line[8],
                    "segmented_answers": [list(jieba.cut(answer)) for answer in answers],
                    "answers": answers,
                    "segmented_question": list(jieba.cut(line[8])),
                    "question_type": "DESCRIPTION",
                    "question_id": line[-1],
                    "fact_or_opinion": "FACT",
                }
                w.write(json.dumps(sample, ensure_ascii=False)+'\n')



#注意：find_fake_answer(sample)是为标准答案找一个假答案，因此测试集如果没答案就用不了
#总之，我发现tensorflow中的dataset.py文件中有对测试集的一个排序（通过question的recall），
#而且，paragraph_extraction文件的策略和这个一样，而preprocess文件只是为了找虚假答案和most_related
#段落，也没再排序啥的，所以说经过这个文件处理后，直接输入模型，test和train的上下文处理方式都不一样。
#而paragraph_extraction正是为了train文件补上这个排序。
# 结论是，test文件基本不用preprocess，paragraph_extraction步骤
if __name__ == '__main__':
    pass

    # transform_dureader_raw(r'D:\Git\decomp\dataset\origin\test.csv',
    #                         '../../data/my/test.json')


    # ../data/demo/trainset/search.train.json
    # with open('../../data/my/train.json','r',encoding='utf-8') as f:
    #     with open('../../data/my/train_pre.json', 'w', encoding='utf-8') as w:
    #         for line in f:
    #             sample = json.loads(line)
    #             find_fake_answer(sample)
    #             w.write(json.dumps(sample, ensure_ascii=False)+'\n')
    #造一个小的数据集测试跑步跑的通
    # i=0
    # with open('../data/du_format/test.json', 'r', encoding='utf-8') as f:
    #     with open('../data/du_format/test_small.json', 'w', encoding='utf-8') as w:
    #         for line in f:
    #             i+=1
    #             if i>32:break
    #             w.write(line)
