# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : utils.py
# Time       ：8/3/2024 9:15 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：several tools
"""
import os
import re
import dgl

import gzip


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import random
import torch
import pickle
import gzip
import shutil
import sys
import datetime
from pyhealth.medcode.codes.atc import ATC
from pyhealth.medcode import InnerMap
from pyhealth.datasets import SampleBaseDataset
from pyhealth.tokenizer import Tokenizer
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime
from itertools import chain
from typing import Optional, Tuple, Union, List
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np

def cal_num(all_samples):
    total_label = []
    for sample in all_samples:
        total_label.append(sample['label'])
    return len(total_label), sum(total_label)



def set_random_seed(seed):
    """ 设置随机种子以确保代码的可重复性 """
    random.seed(seed)       # Python 内置的随机库
    np.random.seed(seed)    # NumPy 库
    torch.manual_seed(seed) # PyTorch 库

    # 如果您使用 CUDA，则还需要添加以下两行代码
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU

def find_prefix_groups(codes, k):
    """
    use for ontology
    """
    cond, proc, drug = codes['conditions'], codes['procedures'], codes['drugs'] # [[]] flattened_list = list(chain.from_iterable(nested_list))
    cond, proc, drug = set().union(*cond), set().union(*proc), []#set().union(*drug) # 不要搞drug
    def get_pair(codes, k):
        groups = {}
        for code in codes:
            prefix = code[:k]  # 获取前k位
            if prefix in groups:
                groups[prefix].append(code)
            else:
                groups[prefix] = [code]
        result = [group for group in groups.values() if len(group) > 1]
        return result
    cond, proc, drug = get_pair(cond, k), get_pair(proc, k), []#get_pair(drug, k)
    prefix = {'conditions': cond, 'procedures': proc, 'drugs': drug}
    return prefix


def extract_ddi(ddi_level, drugs):
    """提取ddi,这里的drug是当前dataset中出现的drug"""
    atc = ATC()
    drugs = set().union(*drugs)
    ddi = atc.get_ddi(gamenet_ddi=True)  # dataframe，这里使用了gamenet的ddi,不要存储
    unique_ddi = list(set().union(*ddi))
    # 这里的convert操作和main没啥问题。
    ddi_atc3 = [
        [ATC.convert(l[0], level=ddi_level), ATC.convert(l[1], level=ddi_level)] for l in unique_ddi if  l[0] in drugs and l[1] in drugs# each row
    ]
    return ddi_atc3



def extract_changes(patient_visits):
    """
    use for transition
    """
    changes = {}
    changes['conditions'], changes['procedures'], changes['drugs'] = [], [], []
    for patient in patient_visits.values():
        cond, proc, drug = patient['conditions'], patient['procedures'], patient['drugs_hist'][1:]# (patient['drugs_hist'] + [patient['drugs']])[1:] # mor，read这些任务不要搞
        for i in range(len(cond) - 1):
            # Convert each visit to a set for easy comparison
            cur_cond, cur_proc, cur_drug = set(cond[i]), set(proc[i]), set(drug[i])
            next_cond, next_proc, next_drug = set(cond[i + 1]), set(proc[i + 1]), set(drug[i + 1])
            # Find the differences between consecutive visits and update the patient_changes
            cha_cond, cha_proc, cha_drug = next_cond^cur_cond, next_proc^cur_proc, next_drug^cur_drug
            changes['conditions'].extend(cha_cond), changes['procedures'].extend(cha_proc), changes['drugs'].extend(cha_drug)
    return changes


def extract_repeats(patient_visits):
    """
    for repeat; 也可以增加倒数第二次的
    """
    changes = {}
    changes['conditions'], changes['procedures'], changes['drugs'] = [], [], []
    for patient in patient_visits.values():

        cond, proc, drug = patient['conditions'], patient['procedures'], patient['drugs']# (patient['drugs_hist'] + [patient['drugs']])[1:] # mor，read这些任务不要搞
        prev_cond, prev_proc, prev_drug = set().union(*cond[:-1]), set().union(*proc[:-1]), set().union(*drug[:-1])
        # print("XXX",cond, drug)
        last_cond, last_proc, last_drug = set(cond[-1]), set(proc[-1]), set(drug[-1])
        rep_cond, rep_proc, rep_drug = last_cond.intersection(prev_cond), last_proc.intersection(prev_proc), last_drug.intersection(prev_drug)
        if rep_cond:
            changes['conditions'].append(list(rep_cond))
        if rep_proc:
            changes['procedures'].append(list(rep_proc))
        if rep_drug:
            changes['drugs'].append(list(rep_drug))

    return changes

def get_tokenizers(dataset, special_tokens=False):
    if not special_tokens:
        special_tokens = ["<unk>"] # 把pad取消
    feature_keys = ["conditions", "procedures", "drugs"]
    feature_tokenizers = {}
    for feature_key in feature_keys:
        feature_tokenizers[feature_key] = Tokenizer(
            tokens=dataset.get_all_tokens(key=feature_key),
            special_tokens=special_tokens,
        )
        print(feature_key, feature_tokenizers[feature_key].get_vocabulary_size())
    return feature_tokenizers



def pad_g_sequence(batch_lis):
    """[torch.stack[torch.tensor],]"""
    # 使用pad_sequence进行填充
    padded_batch = pad_sequence(batch_lis, batch_first=True)
    # 输出结果
    return padded_batch



def pad_batch(embs, lengths):
    """reshape a list into a batch"""
    # Convert lengths to an array if it's not already, for efficient computation
    lengths = np.asarray(lengths)

    # Calculate the cumulative sum of lengths to get the indices for slicing
    cum_lengths = np.cumsum(lengths)
    # lengths = np.cumsum(lengths)
    # batch_lis = []
    # for index, i in enumerate(lengths):
    #     if index == 0:
    #         batch_lis.append(embs[:i, :])
    #     else:
    #         batch_lis.append(embs[lengths[index-1]: lengths[index], :])
    batch_lis = [embs[i - l:i] for i, l in zip(cum_lengths, lengths)]

    batch_lis = torch.nn.utils.rnn.pad_sequence(batch_lis, batch_first=True, padding_value=0) # [torch.randn(4,8), torch.randn(3,8)]->[torch.randn(2,4,8)]
    return batch_lis

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        data = pickle.dump(data, f)
    print("File has beeen saved to {}.".format(file_path))
    return

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def parse_str(template_string):
    """
    Extracts a list of items from a provided string template.
    The function handles different types of delimiters and newline characters.
    template 5还挺难搞的
    """
    # Removing potential leading and trailing brackets and splitting by new line
    sentences = template_string.strip("[]").split('\n')
    if len(sentences) == 1:
        sentences = template_string.strip("[]").split("], [")

    # Removing list item markers and additional brackets, then splitting by comma
    parsed_sentences = [sentence.strip("- ").strip("[]").strip("0123456789. [ ").split(", ") for sentence in sentences if sentence.strip()]

    return parsed_sentences


def process_row(x):
    """处理为三元组"""
    triple_lis = x
    new_df = []
    delet_num = 0
    delet_set = []

    for j in triple_lis:
        if len(j) == 3:
            new_df.append(j)
        elif len(j) in [4, 5, 6, 7, 8, 9]:
            list_items = ['may', 'can', 'is', 'have', 'has', 'be', 'should']
            if any(item in j[-2] for item in list_items):
                new_j = [', '.join(j[:-2]), j[-2], j[-1]]
                new_df.append(new_j)
            elif any(item in j[1] for item in list_items):
                new_j = [j[0], j[1], ', '.join(j[2:])]
                new_df.append(new_j)
            else:
                delet_num += 1
                delet_set.extend(j)
        elif len(j) <= 2:
            delet_num += 1
            delet_set.extend(j)
        elif len(j) > 6 and sum(item.count('[') for item in j) > 0:
            lis = extract_sentences_from_list(j)
            for k in lis:
                if len(k) == 3:
                    new_df.append(k)
                else:
                    delet_num += 1
                    delet_set.extend(j)
        else:
            delet_num += 1
            delet_set.extend(j)

    # print(len(new_df[0]), new_df[0])

    return new_df #, delet_num, delet_set

def extract_sentences_from_list(input_list):
    """
    Extracts and formats sentences from a given list where each sentence is split and scattered.
    The function combines split parts of sentences correctly.
    """
    # Processing the list to extract sentences
    combined_sentences = []
    temp_sentence = []

    for item in input_list:
        # Check for the start and end of a sentence based on brackets
        if item.startswith('['):
            # If there's already a sentence being formed, append it first
            if temp_sentence:
                combined_sentences.append(temp_sentence)
                temp_sentence = []
            # Start a new sentence
            temp_sentence.append(item.strip('[]'))
        elif item.endswith(']'):
            # Complete the sentence
            temp_sentence.append(item.strip('[]'))
            combined_sentences.append(temp_sentence)
            temp_sentence = []
        else:
            # Continue building the sentence
            temp_sentence.append(item)

    # Handling any remaining items in temp_sentence
    if temp_sentence:
        combined_sentences.append(temp_sentence)

    return combined_sentences

def replace_substrings(triple_list, entity_name, index):
    """
    Replace any part of an item in the triple that is a substring of the entity_name with the entity_name itself.
    The comparison is case-insensitive.
    """
    new_triple_list = []
    for triple in triple_list:
        new_triple = []
        for item in triple:
            # Check if the item contains any substring of entity_name
            # print(item.lower())
            item = item.strip("],")
            if item.lower() in entity_name.lower():
                new_triple.append(index)
            else:
                new_triple.append(item)
        new_triple_list.append(new_triple)
    return new_triple_list

def fuzz_match(row):
    """如果有两个字符重叠，则替换"""
    triple_lis = row['processed_triple']
    entity_name = row['entity_name']
    index = row['index']
    new_triple_lis = replace_substrings(triple_lis, entity_name, index)
    return new_triple_lis


def get_atc_name(level):
    """for atc, 这里很奇怪，level为4的话和level为3的设定一致"""
    level = level + 1
    code_sys = ATC(refresh_cache=True)  # 第一次需要
    name_map = {}
    for index in code_sys.graph.nodes:
        if len(index) == level:
            name = code_sys.graph.nodes[index]['name']
            name_map[index] = name
    return name_map



def get_aux_icd(feature_key):
    """有些old icd找不到"""
    if feature_key == 'conditions':
        colspecs = [(0, 5), (6, 14), (15, 16), (17, 77), (78, None)]
        # Read the data into a DataFrame
        df = pd.read_fwf('/home/czhaobo/KnowHealth/data/icd10cm_order_2016.txt', colspecs=colspecs, header=None)
        # Assign column names
        df.columns = ['ID', 'Code', 'Flag', 'Description', 'Add']
        df['Description'] = df['Description'].apply(lambda x: x.split(',')[0])

        df_trimmed = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        dic = df_trimmed.set_index('Code')['Description'].to_dict()
    else:
        colspecs = [(0, 8), (8, None)]
        # Read the data into a DataFrame
        df = pd.read_fwf('/home/czhaobo/KnowHealth/data/icd10pcs_codes_2016.txt', colspecs=colspecs, header=None)
        df2 = pd.read_fwf('/home/czhaobo/KnowHealth/data/icd10pcs_codes_2017.txt', colspecs=colspecs, header=None)
        df3 = pd.read_fwf('/home/czhaobo/KnowHealth/data/icd10pcs_codes_2021.txt', colspecs=colspecs, header=None)
        # Assign column names
        df.columns = ['Code', 'Description']
        df2.columns = ['Code', 'Description']
        df3.columns = ['Code', 'Description']
        df['Description'] = df['Description'].apply(lambda x: x.split(',')[0])
        df2['Description'] = df2['Description'].apply(lambda x: x.split(',')[0])
        df3['Description'] = df3['Description'].apply(lambda x: x.split(',')[0])

        df_trimmed = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df2_trimmed = df2.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df3_trimmed = df3.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        # print(df_trimmed.head())
        dic = df_trimmed.set_index('Code')['Description'].to_dict()
        dic2 = df2_trimmed.set_index('Code')['Description'].to_dict()
        dic3 = df3_trimmed.set_index('Code')['Description'].to_dict()
        dic.update(dic2)
        dic.update(dic3)

    return dic


def get_stand_system(dataset):
    """返回三个编码系统，不然太慢了"""
    if dataset=='MIMIC-III':
        diag_sys = InnerMap.load("ICD9CM")
        proc_sys = InnerMap.load("ICD9PROC")
        med_sys = ATC(refresh_cache=False)
    else:
        diag_sys = InnerMap.load("ICD10CM")
        proc_sys = InnerMap.load("ICD10PROC")
        med_sys = ATC(refresh_cache=False)
    return diag_sys, proc_sys, med_sys


def get_node_name(code_type, reverse_stand=False):
    """for ICD9CM-diag, for ICD9PROC"""
    code_sys = InnerMap.load(code_type)
    name_map = {}
    for index in code_sys.graph.nodes:
        name = code_sys.graph.nodes[index]['name']
        name_map[index] = name
    if reverse_stand:
        name_map = {key.replace('.', ''): value for key, value in name_map.items()}  # [{ATC, name}]
    return name_map


def extract_last_visit(dataset):
    """提取最后一次数据, dataset必须是train dataset"""
    last_visits = {}
    # 遍历记录，更新每个patient_id的最后一次visit记录，确保visit_id以整数形式正确比较
    for record in dataset:
        patient_id = record['patient_id']
        visit_id = int(record['visit_id'])  # 将visit_id转换为整数
        if patient_id not in last_visits or visit_id > int(last_visits[patient_id]['visit_id']):
            last_visits[patient_id] = record
    print("Patient Number: ", len(last_visits))

    extract_info = {}
    extract_info['conditions'] = []
    extract_info['procedures'] = []
    extract_info['drugs'] = []
    for patient_data in last_visits.values():
        cond = patient_data['conditions']
        proc = patient_data['procedures']
        drug = patient_data['drugs'] # (patient_data['drugs_hist'] + [patient_data['drugs']])[1:] # label, 不要搞
        extract_info['conditions'].extend(cond)
        extract_info['procedures'].extend(proc)
        extract_info['drugs'].extend(drug)
    return last_visits, extract_info


def split_by_patient(
        dataset: SampleBaseDataset,
        ratios: Union[Tuple[float, float, float], List[float]],
        train_ratio=1.0,
        seed: Optional[int] = None,
        warm_cold: bool = False,
):
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    patient_indx = list(dataset.patient_to_index.keys()) # 存储数据 {patientID: [index]}
    num_patients = len(patient_indx)
    np.random.shuffle(patient_indx)
    train_patient_indx = patient_indx[: int(num_patients * ratios[0])]
    np.random.seed(seed)
    np.random.shuffle(train_patient_indx)
    train_patient_indx = train_patient_indx[: int(len(train_patient_indx) * train_ratio)]
    val_patient_indx = patient_indx[
                       int(num_patients * ratios[0]): int(
                           num_patients * (ratios[0] + ratios[1]))
                       ]
    test_patient_indx = patient_indx[int(num_patients * (ratios[0] + ratios[1])):]
    train_index = list(
        chain(*[dataset.patient_to_index[i] for i in train_patient_indx])
    )
    val_index = list(chain(*[dataset.patient_to_index[i] for i in val_patient_indx]))
    test_index = list(chain(*[dataset.patient_to_index[i] for i in test_patient_indx]))

    min_length = min(len(lst) for lst in dataset.patient_to_index.values())
    print("最短列表的长度为:", min_length)

    if warm_cold:
        warm_patient_index = []
        cold_patient_index = []
        # 这里放一些东西
        for i in test_patient_indx:
            patient_index = dataset.patient_to_index[i] # lis
            if len(patient_index) > 1: # 最少是1数据来着
                warm_patient_index.extend(patient_index)
            else:
                cold_patient_index.extend(patient_index)
        if warm_cold == 'warm':
            test_dataset = torch.utils.data.Subset(dataset, warm_patient_index)
        elif warm_cold == 'cold':
            test_dataset = torch.utils.data.Subset(dataset, cold_patient_index)
    else:
        test_dataset = torch.utils.data.Subset(dataset, test_index)

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    # test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset


def split_by_patient_one(
        dataset: SampleBaseDataset,
        ratios: Union[Tuple[float, float, float], List[float]],
        train_ratio=1.0,
        seed: Optional[int] = None,
        warm_cold: bool = False,
):
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    patient_indx = list(dataset.patient_to_index.keys()) # 存储数据 {patientID: [index]}
    num_patients = len(patient_indx)
    np.random.shuffle(patient_indx)
    train_patient_indx = patient_indx[: int(num_patients * ratios[0])]
    np.random.seed(seed)
    np.random.shuffle(train_patient_indx)
    train_patient_indx = train_patient_indx[: int(len(train_patient_indx) * train_ratio)]
    val_patient_indx = patient_indx[
                       int(num_patients * ratios[0]): int(
                           num_patients * (ratios[0] + ratios[1]))
                       ]
    test_patient_indx = patient_indx[int(num_patients * (ratios[0] + ratios[1])):]
    train_index = list(
        chain(*[dataset.patient_to_index[i] for i in train_patient_indx])
    )
    val_index = list(chain(*[dataset.patient_to_index[i] for i in val_patient_indx]))
    test_index = list(chain(*[dataset.patient_to_index[i] for i in test_patient_indx]))

    min_length = min(len(lst) for lst in dataset.patient_to_index.values())
    # print("最短列表的长度为:", min_length)

    if warm_cold:
        warm_patient_index = []
        cold_patient_index = []
        # 这里放一些东西
        for i in test_patient_indx:
            patient_index = dataset.patient_to_index[i] # lis
            if len(patient_index) > 1: # 最少是1数据来着
                warm_patient_index.extend(patient_index)
            else:
                cold_patient_index.extend(patient_index)
        test_dataset_warm = torch.utils.data.Subset(dataset, warm_patient_index)
        test_dataset_cold = torch.utils.data.Subset(dataset, cold_patient_index)
    else:
        test_dataset_warm = torch.utils.data.Subset(dataset, test_index)
        test_dataset_cold = None

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    # test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset_warm, test_dataset_cold

def achieve_samples(dataset):
    """subset没有办法获取samples,或者直接重构subset方法
    https://www.cnblogs.com/orion-orion/p/15906086.html"""
    samples = []
    for i in range(len(dataset)):
        samples.append(dataset[i])
    return samples



def near(query_emb, all_reps_emb, index_name, k=5, real_index=True):
    """
    :param query_emb:array
    :param index_name: array, keys of dic
    :param k:
    :return:
    """
    if not isinstance(query_emb, np.ndarray):
        query_emb = query_emb.cpu().detach().numpy()
    dist = cdist(query_emb, all_reps_emb) # 距离
    nn_index_topk = np.argsort(dist, axis=1)[:, :k]
    if real_index:
        index = index_name[nn_index_topk]
    else:
        index = nn_index_topk
    return index



def plot_tsne(embedding_layer, title='t-SNE Visualization'):
    """
    Plot t-SNE visualization of node embeddings for a PyTorch embedding layer.

    Parameters:
        embedding_layer (torch.nn.Embedding): PyTorch embedding layer.
        title (str): Title of the plot (default is 't-SNE Visualization').

    Returns:
        None
    """
    # 获取嵌入层的权重（嵌入矩阵）
    embedding_matrix = embedding_layer.data.cpu().numpy()

    # 使用 sklearn 的 t-SNE 进行降维
    tsne = TSNE(n_components=2)
    embedding_2d = tsne.fit_transform(embedding_matrix)

    # 绘制 t-SNE 图像
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], marker='.')
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    # plt.show()
    plt.savefig('/home/czhaobo/HyperHealth/draw/img/'+ title + '.pdf', format='pdf', dpi=300)


def decompress_gz_files(directory):
    # 获取目录下所有文件
    files = os.listdir(directory)

    # 遍历每个文件
    for file in files:
        # 检查文件是否为.gz文件
        if file.endswith('.gz'):
            file_path = os.path.join(directory, file)
            output_path = os.path.splitext(file_path)[0]  # 去除.gz后缀

            # 打开.gz文件并解压到新文件
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())

            print(f"解压文件 '{file_path}' 到 '{output_path}'")


def f1_cal(pred, ground):
    # 转换为集合用于计算
    pred_set = set(pred)
    ground_set = set(ground)
    # 真正例 (TP): 预测和实际都是正的
    tp = len(pred_set.intersection(ground_set))
    print("TP", pred_set.intersection(ground_set))
    # 假正例 (FP): 预测为正但实际为负
    fp = len(pred_set - ground_set)
    print("FP", pred_set - ground_set)
    # 假负例 (FN): 预测为负但实际为正
    fn = len(ground_set - pred_set)
    print("FN", ground_set - pred_set)
    # 计算精确度和召回率
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    # 计算 F1 分数
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1_score, fp, fn

def jaccard(pred, ground):
    pred_set = set(pred)
    ground_set = set(ground)
    # 计算 Jaccard 相似度
    jaccard = len(pred_set.intersection(ground_set)) / len(pred_set.union(ground_set)) if len(pred_set.union(ground_set)) > 0 else 0
    return jaccard



"""adjust_learning_rate"""
def lr_poly(base_lr, iter, max_iter, power):
    if iter > max_iter:
        iter = iter % max_iter
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, config, max_iter):
    lr = lr_poly(config['LR'], i_iter, max_iter, 0.9) # power=0.9
    optimizer.param_groups[0]['lr'] = np.around(lr,5)
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

if __name__ == '__main__':
    pass
    # node_map = get_atc_name(4)
    # node_map2 = get_atc_name(3)
    # print(len(node_map), len(node_map2))
    # from tes_fun1 import kg, tes
    # print(kg)
    # tes()
    # from tes_fun1 import kg
    # print(kg)

    # dic = get_node_name('ICD9CM', reverse_stand=True) + (get_node_name('ICD9PROC', reverse_stand=True))
    # print(dic)

    # 指定要解压的目录路径
    # directory_path = '/home/czhaobo/HyperHealth/data/physionet.org/files/eicu-crd/2.0'
    # # 调用函数解压目录下的所有.gz文件
    # decompress_gz_files(directory_path)
    # from pyhealth.datasets import SampleEHRDataset
    #
    # root_to = '/home/czhaobo/HyperHealth/data/mortality-gpt/MIII/processed/'
    # samples = load_pickle(root_to + 'datasets_pre_stand.pkl')
    # print("load success!")
    # sample_dataset = SampleEHRDataset(  # 这个贼耗时
    #     samples,
    #     dataset_name="MIII",
    #     task_name='tes',
    # )
    # train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, (0.8, 0.1, 0.1), seed=2023, warm_cold='warm')

    # label = [['B05X', 'A02B', 'C10A', 'B05B', 'C05B', 'A12C', 'V04C', 'N01A', 'N05C', 'N06A', 'A12B']]
    # pred = [['A02B', 'A06A', 'A12C', 'B05B', 'B05X', 'C05B', 'C07A', 'C10A', 'N01A', 'N02B', 'N05C', 'N06A', 'V04C']]
    # con = f1_cal(pred[0],label[0])
    # print(con)
    # hypercare
    # pred = [['A02B', 'A04A', 'A06A', 'A12B', 'A12C', 'B05B', 'B05X', 'C05B', 'N01A', 'N02A', 'N02B', 'N05B', 'N05C', 'V04C']]
    # label =  [['B01A', 'B05X', 'C01B', 'A02B', 'V04C', 'N01A', 'V06D', 'N02B', 'A06A', 'N07B', 'A12C', 'N02A']]
    # # con = f1_cal(pred[0],label[0])
    # con = jaccard(pred[0],label[0])
    # print(con)


    # graphcare
    # pred =[['A02B', 'A04A', 'A06A', 'B03A', 'B02A', 'B05X', 'C05B', 'N01A', 'N02A', 'N02B', 'N05C', 'V04C']]
    # label = [['B01A', 'B05X', 'C01B', 'A02B', 'V04C', 'N01A', 'V06D', 'N02B', 'A06A', 'N07B', 'A12C', 'N02A']]
    # con = f1_cal(pred[0],label[0])
    # print(con)
    # con = jaccard(pred[0],label[0])
    # print(con)

    # pred =[['A02B', 'A04A', 'A06A', 'A12B', 'A12C', 'B05X', 'C05B', 'N01A', 'N02A', 'N02B', 'J05A', 'V04C']]
    # label = [['B01A', 'B05X', 'C01B', 'A02B', 'V04C', 'N01A', 'V06D', 'N02B', 'A06A', 'N07B', 'A12C', 'N02A']]
    # con = f1_cal(pred[0],label[0])
    # print(con)
    # con = jaccard(pred[0],label[0])
    # print(con)



