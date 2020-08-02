#!/usr/bin/env python
#-*- coding:utf-8 -*-

#这个文件是因为原来的数据处理方式不好，尤其是训练集和测试集不一样，那么我就要改变
#dataset.py文件中load_dataset函数的非train部分加以改变
import csv
import json
import re

#将原来用bm25检索的上下文直接拿来用，即取origin文件中的，那么就设置成只有一篇文章
#rc_model.py文件的测试和训练
#超参设置 最大篇章数为1，最大单篇文章的字数。。。
import jieba

def transform_dureader(origin_path, dureader_path, mode = 'train'):
    j=0
    with open(origin_path,'r', encoding='utf-8') as f:
        with open(dureader_path,'w',encoding='utf-8') as w:

            for line in json.load(f)['data']:

                #造一个小数据集方便实验代码是否正常
                # j += 1
                # if j >32:break

                question = line['paragraphs'][0]['qas'][0]['question']

                id = line['paragraphs'][0]['qas'][0]['id']
                context = line['paragraphs'][0]['context']
                #由于只取每个文档的most_related_paragraph这一句，我干脆整成一句
                paragraphs = [context]
                # 测试集中有的不一定找得到答案,不一定找得到相关段落
                if mode == 'train':
                    answer = line['paragraphs'][0]['qas'][0]['answers'][0]['text']
                    start_id = line['paragraphs'][0]['qas'][0]['answers'][0]['answer_start']



                if mode == 'train':
                    # 经过bm25检索之后，相当于只有一个文档
                    documents = [{
                        'is_selected': True,
                        # 'title':title,
                        "most_related_para": 0,
                        'paragraphs': paragraphs,
                        # 'segmented_title': list(jieba.cut(title)),
                        'segmented_paragraphs': [list(jieba.cut(para)) for para in paragraphs],
                    }]
                    sample = {
                        'documents':documents,
                        "answer_spans": [[start_id, start_id + len(answer)]],
                        "question": question,
                        "segmented_answers": list(jieba.cut(answer)),
                        "answers": answer,
                        "answer_docs": [0],
                        "segmented_question": list(jieba.cut(question)),
                        "question_type": "DESCRIPTION",
                        "question_id": id,
                        "fact_or_opinion": "FACT",
                    }
                else:
                    documents = [{
                        'is_selected': True,
                        # 'title':title,
                        'paragraphs': paragraphs,
                        # 'segmented_title': list(jieba.cut(title)),
                        'segmented_paragraphs': [list(jieba.cut(para)) for para in paragraphs],
                    }]
                    sample = {
                        'documents': documents,
                        "question": question,
                        "segmented_question": list(jieba.cut(question)),
                        "question_type": "DESCRIPTION",
                        "question_id": id,
                        "fact_or_opinion": "FACT",
                    }
                w.write(json.dumps(sample, ensure_ascii=False)+'\n')

transform_dureader(r'D:\Git\decomp\processed\origin\train.json',
                   'D:/Git/DuReader/data/my_preprocess/train.json',
                   mode = 'train')
#                    训练集和测试集的主要区别在于是否有相关段落，答案，答案开始标签，answer_docs