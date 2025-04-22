# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : data.py
# Time       ：8/3/2024 9:13 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：设定task
"""
import collections
import faiss
import dgl
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pyhealth.data import Patient, Visit
from torch.utils.data import DataLoader
from config import config
from utils import load_pickle, find_prefix_groups, extract_changes, extract_ddi, extract_repeats, get_atc_name, get_node_name, near
from transformers import AutoTokenizer, AutoModel
from scipy.sparse import csr_matrix
from pyhealth.medcode import ATC
from itertools import combinations
from utils import save_pickle, load_pickle, extract_last_visit

from pyhealth.datasets import SampleBaseDataset
from pyhealth.datasets.utils import list_nested_levels, flatten_list
from typing import Dict, List


class SampleEHRDatasetSIMPLE(SampleBaseDataset):
    def __init__(self, samples: List[str], code_vocs=None, dataset_name="", task_name="", all=False):
        super().__init__(samples, dataset_name, task_name)
        self.samples = samples
        if all:
            self.input_info: Dict = self._validate()


    @property
    def available_keys(self) -> List[str]:
        """Returns a list of available keys for the dataset.

        Returns:
            List of available keys.
        """
        keys = self.samples[0].keys()
        return list(keys)

    def _validate(self) -> Dict:
        """ 1. Check if all samples are of type dict. """
        keys = self.samples[0].keys()

        """
        4. For each key, check if it is either:
            - a single value
            - a single vector
            - a list of codes
            - a list of vectors
            - a list of list of codes
            - a list of list of vectors
        Note that a value is either float, int, or str; a vector is a list of float 
        or int; and a code is str.
        """
        # record input information for each key
        input_info = {}
        for key in keys:
            """
            4.1. Check nested list level: all samples should either all be
            - a single value (level=0)
            - a single vector (level=1)
            - a list of codes (level=1)
            - a list of vectors (level=2)
            - a list of list of codes (level=2)
            - a list of list of vectors (level=3)
            """
            levels = set([list_nested_levels(s[key]) for s in self.samples[:5]]) # 只取前5个判断足够

            level = levels.pop()[0]

            # flatten the list
            if level == 0:
                flattened_values = [s[key] for s in self.samples]
            elif level == 1:
                flattened_values = [i for s in self.samples for i in s[key]]
            elif level == 2:
                flattened_values = [j for s in self.samples for i in s[key] for j in i]
            else:
                flattened_values = [
                    k for s in self.samples for i in s[key] for j in i for k in j
                ]

            """
            4.2. Check type: the basic type of each element should be float, 
            int, or str.
            """
            types = set([type(v) for v in flattened_values[:5]]) # 只取前5个判断足够
            type_ = types.pop()
            """
            4.3. Combined level and type check.
            """
            if level == 0:
                # a single value
                input_info[key] = {"type": type_, "dim": 0}
            elif level == 1:
                # a single vector or a list of codes
                if type_ in [float, int]:
                    # a single vector
                    lens = set([len(s[key]) for s in self.samples])
                    input_info[key] = {"type": type_, "dim": 1, "len": lens.pop()}
                else:
                    # a list of codes
                    # note that dim is different from level here
                    input_info[key] = {"type": type_, "dim": 2}
            elif level == 2:
                # a list of vectors or a list of list of codes
                if type_ in [float, int]:
                    lens = set([len(i) for s in self.samples for i in s[key]])
                    input_info[key] = {"type": type_, "dim": 2, "len": lens.pop()}
                else:
                    # a list of list of codes
                    # note that dim is different from level here
                    input_info[key] = {"type": type_, "dim": 3}
            else:
                # a list of list of vectors
                lens = set([len(j) for s in self.samples for i in s[key] for j in i])
                input_info[key] = {"type": type_, "dim": 3, "len": lens.pop()}

        return input_info

    def __len__(self):
        return len(self.samples)



def categorize_los(days: int):
    """Categorizes length of stay into 10 categories.

    One for ICU stays shorter than a day, seven day-long categories for each day of
    the first week, one for stays of over one week but less than two,
    and one for stays of over two weeks.

    Args:
        days: int, length of stay in days

    Returns:
        category: int, category of length of stay
    """
    # ICU stays shorter than a day
    if days < 1:
        return 0
    # each day of the first week
    elif 1 <= days <= 7:
        return days
    # stays of over one week but less than two
    elif 7 < days <= 14:
        return 8
    # stays of over two weeks
    else:
        return 9


def length_of_stay_prediction_mimic3_fn(patient: Patient):
    """Processes a single patient for the mortality prediction task.
    """
    samples = []
    for i in range(len(patient)): # visit 次数
        visit: Visit = patient[i]

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs], # 实际不需要使用
                "drugs_hist": [drugs],
                "conditions_kg": [extract_subgraph(conditions,fuzz=True, type='ICD9')],
                "procedures_kg": [extract_subgraph(procedures,fuzz=True, type='ICD9')],
                "drugs_hist_kg": [extract_subgraph(drugs,fuzz=True, type='ATC')],
                "label": los_category,
            }
        )


    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]
    samples[0]["conditions_kg"] = samples[0]["conditions_kg"]#[extract_subgraph(tmp_samples[0]["conditions"])] # 不然会更新错误
    samples[0]["procedures_kg"] = samples[0]["procedures_kg"]#[extract_subgraph(tmp_samples[0]["procedures"])]
    samples[0]["drugs_hist_kg"] = samples[0]["drugs_hist_kg"]#[extract_subgraph(tmp_samples[0]["drugs_hist"])]

    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]

        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]

        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]

        samples[i]["conditions_kg"] = samples[i - 1]["conditions_kg"] + samples[i]["conditions_kg"]

        samples[i]["procedures_kg"] = samples[i - 1]["procedures_kg"] + samples[i]["procedures_kg"]

        samples[i]["drugs_hist_kg"] = samples[i - 1]["drugs_hist_kg"] + samples[i]["drugs_hist_kg"]

    return samples


def length_of_stay_prediction_mimic4_fn(patient: Patient):
    """Processes a single patient for the mortality prediction task.
    """
    samples = []
    for i in range(len(patient)): # visit 次数
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "drugs_hist": [drugs],
                "conditions_kg": [extract_subgraph(conditions,fuzz=True, type='ICD10')],
                "procedures_kg": [extract_subgraph(procedures,fuzz=True, type='ICD10')],
                "drugs_hist_kg": [extract_subgraph(drugs,fuzz=True, type='ATC')],
                "label": los_category,
            }# 这里也可以加入labels_kg，然后利用这个label kg进行计算
        )

    if len(samples) < 2: # [{},{}]
        return samples

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]
    samples[0]["conditions_kg"] = samples[0]["conditions_kg"]
    samples[0]["procedures_kg"] = samples[0]["procedures_kg"]#[extract_subgraph(tmp_samples[0]["procedures"])]
    samples[0]["drugs_hist_kg"] = samples[0]["drugs_hist_kg"]#[extract_subgraph(tmp_samples[0]["drugs_hist"])]

    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]

        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]

        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]

        samples[i]["conditions_kg"] = samples[i - 1]["conditions_kg"] + samples[i]["conditions_kg"]

        samples[i]["procedures_kg"] = samples[i - 1]["procedures_kg"] + samples[i]["procedures_kg"]

        samples[i]["drugs_hist_kg"] = samples[i - 1]["drugs_hist_kg"] + samples[i]["drugs_hist_kg"]

    return samples





def collate_fn_dict(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}


def get_dataloader(dataset, batch_size, shuffle=False, drop_last=False):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_dict,
        drop_last=drop_last
    )

    return dataloader



def load_kg(file_name, index_name, rel_name):
    """read kg file, 这里的前提是需要kg file的对应表已经获得"""
    file = load_pickle(file_name)
    ehr_id = load_pickle(index_name)
    rel_id = load_pickle(rel_name)

    u = file['cui1'].values.tolist()
    v = file['cui2'].values.tolist()
    e = file['rel'].values.tolist()

    # 无向图
    src = u + v
    dst = v + u
    e = e + e # 这里思考一下，是否需要把边的方向也考虑进去

    graph = dgl.graph((src, dst))
    graph.edata['etype'] = torch.tensor(e, dtype=torch.long)


    # 加载fea，暂时随机初始化
    n_fea = torch.randn(len(ehr_id), 2)
    e_fea = torch.randn(len(rel_id), config['DIM'])
    e_fea = e_fea[e]
    graph.ndata['fea'] = n_fea
    graph.edata['fea'] = e_fea

    # self loop
    self_loop_features = torch.zeros(len(ehr_id), 2) # 反正不用的啦
    graph.add_edges(torch.arange(len(ehr_id)), torch.arange(len(ehr_id)), data={'edge_feature': self_loop_features})

    # 无向
    # graph = dgl.to_bidirected(graph, copy_ndata=True)

    print("Load KG success!")
    print("Load KG vocabulary success!")

    return graph, ehr_id, rel_id



def load_kg_after(file_name, index_name, rel_name):
    """read kg file, 这里的前提是需要kg file的对应表已经获得"""
    # file = load_pickle(file_name)
    ehr_id = load_pickle(index_name)
    rel_id = load_pickle(rel_name)


    print("Load KG success!")
    print("Load KG vocabulary success!")

    return [], ehr_id, rel_id


def load_code_convert(dataset='MIII', samples=None):
    """load code convert,必须要在子图采集之后进行"""
    from pyhealth.medcode import CrossMap
    if dataset == 'MIII':
        cm_mapping = CrossMap("ICD9CM", "CCSCM").mapping
        proc_mapping = CrossMap("ICD9PROC", "CCSPROC").mapping
    else:
        cm_mapping = CrossMap("ICD10CM", "CCSCM").mapping # 这里不对，我记得是9，10都有
        proc_mapping = CrossMap("ICD10PROC", "CCSPROC").mapping
    cm_mapping = {key.replace('.', ''): value for key, value in cm_mapping.items()}
    proc_mapping = {key.replace('.', ''): value for key, value in proc_mapping.items()}
    for sample in samples:
        # conditions
        conditions = sample['conditions']
        conditions = [list(map(lambda item: cm_mapping.get(item, [item])[0], sublist)) for sublist in conditions]
        sample['conditions'] = conditions
        # procedures
        procedures = sample['procedures']
        procedures = [list(map(lambda item: proc_mapping.get(item, [item])[0], sublist)) for sublist in procedures]

        sample['procedures'] = procedures
    return samples

def load_code_convert_hyper(dataset='MIII', sample=None):
    """load code convert,必须要在子图采集之后进行"""
    from pyhealth.medcode import CrossMap
    if dataset == 'MIII':
        cm_mapping = CrossMap("ICD9CM", "CCSCM").mapping
        proc_mapping = CrossMap("ICD9PROC", "CCSPROC").mapping
    else:
        cm_mapping = CrossMap("ICD10CM", "CCSCM").mapping # 这里不对，我记得是9，10都有
        proc_mapping = CrossMap("ICD10PROC", "CCSPROC").mapping
    cm_mapping = {key.replace('.', ''): value for key, value in cm_mapping.items()}
    proc_mapping = {key.replace('.', ''): value for key, value in proc_mapping.items()}

    conditions = sample['conditions']
    print("Mapping before, ", conditions[0])
    conditions = [list(map(lambda item: cm_mapping.get(item, [item])[0], sublist)) for sublist in conditions]
    sample['conditions'] = conditions
    print("Mapping after, ", conditions[0])

    # procedures
    procedures = sample['procedures']
    procedures = [list(map(lambda item: proc_mapping.get(item, [item])[0], sublist)) for sublist in procedures]
    sample['procedures'] = procedures

    return sample

def add_edges(dataset, kg, ehr_id_dic):
    """增加ehr边和DDI边, dataset 应该都是train_dataset，避免信息泄漏; 暂时不用"""
    dataset = extract_last_visit(dataset)
    # 加载数据DDI
    atc = ATC()
    ddi = atc.get_ddi(gamenet_ddi=True)  # dataframe，这里使用了gamenet的ddi,不要存储
    unique_ddi = list(set().union(*ddi))
    # 这里的convert操作和main没啥问题。
    ddi_atc3 = [
        [ATC.convert(l[0], level=config['ATCLEVEL']), 'ddi', ATC.convert(l[1], level=config['ATCLEVEL'])] for l in ddi  # each row
    ]
    print(ddi_atc3)

    # 加载EHR数据
    cond, proc, drug = dataset['conditions'], dataset['procedures'], dataset['drugs']
    ehr_cond, ehr_proc, ehr_drug = [], [], []
    def get_ehr(ehr, ehr_lis):
        for patient in ehr_lis:
            for visit_lis in patient:
                pairs = list(combinations(visit_lis, 2))  # [()]
                ehr.extend(pairs)
        return ehr
    cond = get_ehr(ehr_cond, cond)
    proc = get_ehr(ehr_proc, proc)
    drug = get_ehr(ehr_drug, drug)
    unique_cond, unique_proc, unique_drug = list(set().union(*cond)), list(set().union(*proc)), list(set().union(*drug))
    unique_drug.extend(unique_ddi).extend(unique_cond).extend(unique_proc) # [A1001]

    unique_condp, unique_procp, unique_drugp = list(set(cond)), list(set(proc)), list(set(drug))
    unique_condp = [[l[0], 'ehr', l[1]] for l in unique_condp]
    unique_procp = [[l[0], 'ehr', l[1]] for l in unique_procp]
    unique_drugp = [[l[0], 'ehr', l[1]] for l in unique_drugp]
    unique_drugp.extend(ddi_atc3).extend(unique_condp).extend(unique_procp)

    # 重新编码 (重编ID, EHRID)
    max_ehr_id = max(ehr_id_dic.values())
    for ehr in unique_drug:
        if ehr not in ehr_id_dic:
            max_ehr_id += 1  # id增加1
            ehr_id_dic[ehr] = max_ehr_id  # 在dict中新增一个ehr:id

    add_kg = [[ehr_id_dic[triple[0]], triple[1], ehr_id_dic[triple[2]]] for triple in unique_drugp]
    kg.extend(add_kg)

    save_pickle(kg, 'dir')
    save_pickle(kg, 'dir')
    print("KG Done!")
    return


def create_hyperg(dataset, tokenizers, ono_level=2, mapping=False):
    """load hypergraph
    n_nodes: 节点数: {}
    """
    print("Now extract Hyperg data.")
    cond_tokenizer, proc_tokenizer, drug_tokenizer = tokenizers['conditions'], tokenizers['procedures'], tokenizers['drugs']
    num_cond, num_proc, num_drug = cond_tokenizer.get_vocabulary_size(), proc_tokenizer.get_vocabulary_size(), drug_tokenizer.get_vocabulary_size()

    last_visits, extract_info = extract_last_visit(dataset) # {patientid:{cond:[[]]}}, {cond:[[]]}

    # 获取train_bundle, 存储所有hyper edge关系
    cond_train_bundle, proc_train_bundle, drug_train_bundle = [], [], []
    cate_con, cate_proc, cate_drug = [], [], [] # 记录超边类型
    # set rule
    cond_set_bundle, proc_set_bundle, drug_set_bundle = extract_info['conditions'], extract_info['procedures'], extract_info['drugs']
    # print("Sample set, ", cond_set_bundle[:2])

    cond_train_bundle.extend(cond_set_bundle), proc_train_bundle.extend(proc_set_bundle), drug_train_bundle.extend(drug_set_bundle)
    cate_con.extend([0]*len(cond_set_bundle)), cate_proc.extend([0]*len(proc_set_bundle)), cate_drug.extend([0]*len(drug_set_bundle))
    # trans_changes = extract_changes(last_visits)
    # cond_tran_bundle, proc_tran_bundle, drug_tran_bundle = trans_changes['conditions'], trans_changes['procedures'], trans_changes['drugs']
    # cond_train_bundle.extend(cond_tran_bundle), proc_train_bundle.extend(proc_tran_bundle), drug_train_bundle.extend(drug_tran_bundle)
    # cate_con.extend([1]*len(cond_tran_bundle)), cate_proc.extend([1]*len(proc_tran_bundle)), cate_drug.extend([1]*len(drug_tran_bundle))
    #
    # # DDI rule，
    # drug_ddi_bundle = extract_ddi(config['ATCLEVEL'], extract_info['drugs'])
    # drug_train_bundle.extend(drug_ddi_bundle)
    # cate_drug.extend([4] * len(drug_ddi_bundle))


    # repeat rule
    rep_changes = extract_repeats(last_visits)
    cond_rep_bundle, proc_rep_bundle, drug_rep_bundle = rep_changes['conditions'], rep_changes['procedures'], rep_changes['drugs']
    # print("Sample rep, ", cond_rep_bundle[:2])
    cond_train_bundle.extend(cond_rep_bundle), proc_train_bundle.extend(proc_rep_bundle), # drug_train_bundle.extend(drug_rep_bundle)
    cate_con.extend([2]*len(cond_rep_bundle)), cate_proc.extend([2]*len(proc_rep_bundle)), # cate_drug.extend([2]*len(drug_rep_bundle))

    # ontology rule
    ont = find_prefix_groups(extract_info, ono_level)
    cond_ont_bundle, proc_ont_bundle, drug_ont_bundle = ont['conditions'], ont['procedures'], ont['drugs']
    # print("Sample ont, ", cond_ont_bundle[:2])
    cond_train_bundle.extend(cond_ont_bundle), proc_train_bundle.extend(proc_ont_bundle), # drug_train_bundle.extend(drug_ont_bundle)
    cate_con.extend([3]*len(cond_ont_bundle)), cate_proc.extend([3]*len(proc_ont_bundle)), # cate_drug.extend([3]*len(drug_ont_bundle))


    # tokenizer
    # print(cond_train_bundle[0])
    # print(proc_train_bundle[0])
    # print(drug_train_bundle[0])

    # if mapping: # 似乎不用，这里的hyper本身就是源自sample dataset
    #     tmp = load_code_convert_hyper(config['DATASET'], {'conditions': cond_train_bundle, 'procedures': proc_train_bundle, 'drugs': drug_train_bundle})
    #     cond_train_bundle, proc_train_bundle, drug_train_bundle = tmp['conditions'], tmp['procedures'], tmp['drugs']
    #     print("Hyper Mapping success!")

    cond_train_bundle, proc_train_bundle, drug_train_bundle = cond_tokenizer.batch_encode_2d(cond_train_bundle, padding=False), proc_tokenizer.batch_encode_2d(proc_train_bundle, padding=False), drug_tokenizer.batch_encode_2d(drug_train_bundle, padding=False)

    # 构建超图
    print("Now construct Hyperg.")

    def get_bundle(train_bundle, num_node, type_lis):
        indptr, indices, data, typs, hyps = [0], [], [], [], []  # 每行非0元素的起始位置；非0元素的列索引；数据
        for j in range(len(train_bundle)):
            bundle = np.unique(train_bundle[j])
            length = len(bundle)  # 4
            hyp = [j] * length
            typ = [type_lis[j]] * length
            s = indptr[-1]  # 0
            indptr.append((s + length))
            indices.extend(bundle)  # [1,2,3,4,1,2,3,4]
            data.extend([1] * length)
            typs.extend(typ)  # [0,0,0,0,1,1,1,1]
            hyps.extend(hyp)  # [0,0,0,0,1,1,1,1]
           
        H_T = csr_matrix((data, indices, indptr), shape=(len(train_bundle), num_node)).tocoo()  # 超边，节点
        pair_hyper_ehr = list(zip(hyps, H_T.col, typs))  # hyper_edge, node, type
        return H_T, pair_hyper_ehr, num_node # len(H_T.col)

    H_T_cond, pair_cond, n_nodes_cond = get_bundle(cond_train_bundle, num_cond, cate_con) # 毗邻矩阵和节点对
    H_T_proc, pair_proc, n_nodes_proc = get_bundle(proc_train_bundle, num_proc, cate_proc)
    H_T_drug, pair_drug, n_nodes_drug = get_bundle(drug_train_bundle, num_drug, cate_drug)

    return (H_T_cond, H_T_proc, H_T_drug), (pair_cond, pair_proc, pair_drug), (n_nodes_cond, n_nodes_proc, n_nodes_drug)


kg, ehr_id_dic, rel_dic = load_kg_after(config['KG_DATADIR'] + config['DATASET'] + '/triples_id.pkl',
                        config['KG_DATADIR'] + config['DATASET'] + '/ehr_id_map.pkl',
                        config['KG_DATADIR'] + config['DATASET'] + '/rel_id_map.pkl') # 这里需要修改；
max_ehr_id = max(ehr_id_dic.values())
max_rel_id = max(rel_dic.values())
ehr_code2name = {
    'ATC': get_atc_name(config['ATCLEVEL']),
    'ICD9': {**get_node_name('ICD9CM', reverse_stand=True), **get_node_name('ICD9PROC', reverse_stand=True)},
    # 'ICD10CM': get_node_name('ICD10CM', reverse_stand=True).update(get_node_name('ICD9CM', reverse_stand=True)), # 因为他需要用到ICD9CM的名字
    'ICD10': {**get_node_name('ICD10PROC', reverse_stand=True), **get_node_name('ICD10CM', reverse_stand=True), **get_node_name('ICD9CM', reverse_stand=True), **get_node_name('ICD9PROC', reverse_stand=True)},
}

# tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
# model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to('cuda:6') # 注意query nn在一个频道
# index_name_dict, all_emb = (load_pickle(config['KG_DATADIR'] + config['DATASET'] + '/ehr_name_map.pkl'),
#                                  load_pickle(config['KG_DATADIR'] + config['DATASET'] + '/entity_emb.pkl'))
# index_name, part = np.array(list(index_name_dict.keys())), round(all_emb.shape[0]//500) # 这玩意巨耗时间
# faiss_index = faiss.IndexFlatL2(768)  # 使用L2距离
# faiss_index.add(all_emb)  # 向索引中添加数据库向量

def query_nn_neighbor(query, k=5, trun_ratio=0.01):
    """
    :param query: string
    :param file:
    :return:
    """
    # trun_num = round(all_emb.shape[0] * trun_ratio)
    # shuffle_lis = np.random.permutation(all_emb.shape[0])[:trun_num] # 这玩意也巨耗时
    random_int = np.random.randint(0, 100)
    index_n, all_reps_emb = index_name[random_int*part:(random_int+1)*part], all_emb[random_int*part:(random_int+1)*part]
    # index_n, all_reps_emb = index_name[shuffle_lis], all_emb[shuffle_lis]

    query_tokens = tokenizer.batch_encode_plus(query,
                                       padding="max_length",
                                       max_length=25,
                                       truncation=True,
                                       return_tensors="pt")
    # print(query_tokens)
    query_tokens = {k: v.to('cuda:4') for k, v in query_tokens.items()}
    query_cls_rep = model(**query_tokens)[0][:, 0, :] # use cls as repr
    index = near(query_cls_rep, all_reps_emb, index_n, k=k)
    # _, topk = faiss_index.search(query_cls_rep.cpu().detach().numpy(), k)
    return index




def extract_subgraph(code_lis, label=False, fuzz=False, type=None):
    # 检索index
    if fuzz:
        if type:
            code2name = ehr_code2name[type]
            # code_name = [code2name[code] for code in code_lis if code in code2name]
            code_name = [code2name.get(code, code) for code in code_lis] # 没有code则返回其本身
        else:
            code_name = code_lis
        index = query_nn_neighbor(code_name, k=config['FUZZ'], trun_ratio=0.001)
        code_list = index.flatten().tolist()
        code_list = [ehr_id_dic[code] for code in code_list]
    else:
        code_list = []
        for code in code_lis:
            try:
                index = ehr_id_dic[code] # 这里的code本身是没有stand过的
                code_list.append(index)
            except:
                continue
    if label:
        if len(code_list) < 1:
            return [str(max_ehr_id+1)]
        code_list = [str(code) for code in code_list]
        return code_list

    if len(code_lis) <1: # 无法提取子图
        return [[max_ehr_id+1,max_rel_id+1,max_ehr_id+1]]
    etype_all = kg.edata['etype']
    results = []

    for i in range(config['RANDOM_NUM']):
        neighbors, edge_ids, _ = dgl.sampling.random_walk(kg, code_list, length=config['RANDOM_LENGTH'], return_eids=True) # node type
        if neighbors.shape[0] < 1:
            return [[max_ehr_id + 1, max_rel_id + 1, max_ehr_id + 1]]
        neighbors_slice1, neighbors_slice2, edge_slice = neighbors[:,:-1], neighbors[:, 1:], edge_ids[:, :]
        etype = etype_all[edge_slice.reshape(-1)]
        result_tuple = torch.stack((neighbors_slice1.reshape(-1), etype, neighbors_slice2.reshape(-1))).unbind(1)
        result_tuple = torch.stack(result_tuple).numpy().tolist()
        # result_tuple = list(result_tuple).numpy().tolist()
        results.extend(result_tuple)

    return results


