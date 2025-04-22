# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : main_los.py
# Time       ：1/4/2024 8:19 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：
"""


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import json
import time
import pickle
import torch
import numpy as np
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset, SampleEHRDataset
from utils import split_by_patient, load_pickle, save_pickle, set_random_seed, get_tokenizers, achieve_samples
from data import SampleEHRDatasetSIMPLE, length_of_stay_prediction_mimic3_fn, length_of_stay_prediction_mimic4_fn, get_dataloader, create_hyperg,  ehr_id_dic, rel_dic, load_code_convert
from trainer import Trainer
from config import config
from models_los import HyperRec
# from memory_profiler import profile

set_random_seed(528) # 换成575试试，因为575的更多191； 603更少182个

import torch.nn as nn
from itertools import repeat


def re_generate_dataset(samples, seed):
    sample_dataset = SampleEHRDataset(  # 这个贼耗时
        samples,
        dataset_name=config['DATASET'],
        task_name=config['TASK'],
    )  # convert_dataset(samples, all_samples=True) # 这一步其实最耗时

    # root_to = '/home/czhaobo/HyperHealth/data/drugrec/{}/processed/'.format(config['DATASET'])
    #  split & save dataset
    train_dataset,_, test_dataset = split_by_patient(
        sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
        train_ratio=0.001,  # Train test split
        seed=seed,
    )
    # train_samples, test_samples = achieve_samples(train_dataset),  achieve_samples(test_dataset)
    # save_pickle(train_samples, root_to + 'train_samples.pkl')
    # save_pickle(val_samples, root_to + 'val_samples.pkl')
    # save_pickle(test_samples, root_to + 'test_samples.pkl')
    print("Regerenate dataset done!")
    return train_dataset, test_dataset

def convert_dataset(samples, all_samples=False):
    """避免繁琐的处理"""
    return SampleEHRDatasetSIMPLE(
                    samples,
                    dataset_name=config['DATASET'],
                    task_name=config['TASK'],
                    all=all_samples,
                )

# @profile(precision=4, stream=open("memory_profiler.log", "w+"))
def run_single_config():
    a = torch.ones((10000, 30000)).to('cuda:' + config['GPU'])

    # load datasets
    # STEP 1: load data
    task = 'no_core_set'
    if task == 'core_set':
        """使用核心集，过滤后的, 建议重新搞一个文件夹，不要使用pyhealth"""
        root_to = '/home/czhaobo/HyperHealth/data/drugrec/{}/processed/'.format(config['DATASET'])
        if not os.path.exists(root_to + 'datasets_pre.json'):
            json_path = root_to + 'datasets.json'
            with open(json_path, 'r') as file:
                base_dataset = json.load(file)
            samples = []
            for patient_id, patient in enumerate(base_dataset):
                samples.extend(length_of_stay_prediction_mimic3_fn(patient))
            with open(root_to + 'datasets_pre.json', 'w') as file:
                json.dump(samples, file, indent=4)
            print("initial dataset done!")
        else:
            with open(root_to + 'datasets_pre.json', 'r') as file:
                samples = json.load(file)
            print("Load dataset done!")

        sample_dataset = SampleEHRDataset(
            samples,
            dataset_name='MIMIC3',
            task_name='los_pred',
        )
        print('dataset done!')
    else:
        root_to = '/home/czhaobo/HyperHealth/data/los-gpt/{}/processed/'.format(config['DATASET'])
        if not os.path.exists(root_to + 'datasets_pre_stand.pkl'):
            if config['DATASET'] == 'MIII':
                base_dataset = MIMIC3Dataset(
                    root="/home/czhaobo/HyperHealth/data/physionet.org/files/mimiciii/1.4",
                    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
                    code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})}, # 这里graphcare的ATC-level是3；和我们在data阶段有差别
                    dev=False,
                    refresh_cache=False,
                )
                base_dataset.stat()

                # set task
                sample_dataset = base_dataset.set_task(length_of_stay_prediction_mimic3_fn) # 按照task_fn进行处理
                sample_dataset.stat()
            else:
                base_dataset = MIMIC4Dataset(
                    root="/home/czhaobo/HyperHealth/data/physionet.org/files/mimiciv/2.0/hosp",
                    tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                    code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})}, # 4
                    dev=False,
                    refresh_cache=False,
                )
                base_dataset.stat()
                # set task
                sample_dataset = base_dataset.set_task(length_of_stay_prediction_mimic4_fn)
                sample_dataset.stat()

            #  split & save dataset
            train_dataset, val_dataset, test_dataset = split_by_patient(
                sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
                train_ratio=1.0,  # Train test split
                seed=528
            )

            samples = sample_dataset.samples
            train_samples, val_samples, test_samples = achieve_samples(train_dataset), achieve_samples(val_dataset), achieve_samples(test_dataset)
            save_pickle(samples, root_to + 'datasets_pre_stand.pkl')
            save_pickle(train_samples, root_to + 'train_samples.pkl')
            save_pickle(val_samples, root_to + 'val_samples.pkl')
            save_pickle(test_samples, root_to + 'test_samples.pkl')

            # HyperG
            tokenizers = get_tokenizers(sample_dataset)
            hyperg_tup, pair_tup, n_nodes_tup = create_hyperg(train_dataset, tokenizers, ono_level=2) # 前两位相等即可
            save_pickle(hyperg_tup, root_to + 'hyperg_tup.pkl')
            save_pickle(pair_tup, root_to + 'pair_tup.pkl')
            save_pickle(n_nodes_tup, root_to + 'n_nodes_tup.pkl')

            print("initial dataset done!")
            print("Please run again!")
            return
        else:
            start = time.time()
            samples = load_pickle(root_to + 'datasets_pre_stand.pkl')
            # train_samples = load_pickle(root_to + 'train_samples.pkl')
            # val_samples = load_pickle(root_to + 'val_samples.pkl')
            # test_samples = load_pickle(root_to + 'test_samples.pkl')


            # few-shot怕是需要，因为训练不够 # 全量解锁，使用更少的embedding
            # train_samples = load_code_convert(dataset=config['DATASET'], samples=train_samples) # 很有用 感觉去掉算了
            # val_samples = load_code_convert(dataset=config['DATASET'], samples=val_samples) # 很有用 感觉去掉算了
            # test_samples = load_code_convert(dataset=config['DATASET'], samples=test_samples) # 很有用 感觉去掉算了
            end = time.time()
            print("Load data done! Cost time {} s".format(end-start))

            if config['DEV']: # 加入超图后feature_token initial 有问题哈
                print("DEV train mode: 1000 patient")
                samples = samples[:3000]
                sample_dataset = SampleEHRDataset( # 这个贼耗时
                    samples,
                    dataset_name=config['DATASET'],
                    task_name=config['TASK'],
                ) # convert_dataset(samples, all_samples=True) # 这一步其实最耗时
                train_dataset, val_dataset, test_dataset = split_by_patient(
                    sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
                    train_ratio=1.0,  # Train test split
                    seed=528
                )
                del samples
                # 需要重新生成HyperG
                tokenizers = get_tokenizers(sample_dataset) # 删减hyper edge
                hyperg_tup, pair_tup, n_nodes_tup = create_hyperg(train_dataset, tokenizers, ono_level=2) # 前两位相等即可
                print("Pair HyperEdge",hyperg_tup[0].shape, hyperg_tup[1].shape, hyperg_tup[2].shape)
                print("Pair HyperEdge",len(pair_tup[0]), len(pair_tup[1]), len(pair_tup[2]))

                load_dataset = time.time()
                print('Dataset done!, Cost {} s'.format(load_dataset - end))
            else:
                sample_dataset = convert_dataset(samples, all_samples=True)
                # train_dataset, val_dataset, test_dataset = split_by_patient(
                #     sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
                #     train_ratio=1.0,  # Train test split
                #     seed=528
                # ) # 这样似乎更快，固定随机种子的时候是一样的；

                # train_dataset = convert_dataset(train_samples)
                # val_dataset = convert_dataset(val_samples)
                # test_dataset = convert_dataset(test_samples)

                train_dataset, test_dataset = re_generate_dataset(samples, 572) # 重新生成mo

                del samples # , train_samples, test_samples
                endt = time.time()
                print('Train Dataset done!, Cost {} s'.format(endt - end))

                # # load HyperG
                # hyperg_tup = load_pickle(root_to + 'hyperg_tup.pkl')
                # pair_tup = load_pickle(root_to + 'pair_tup.pkl')
                # n_nodes_tup = load_pickle(root_to + 'n_nodes_tup.pkl')
                # print("Pair HyperEdge", len(pair_tup[0]), len(pair_tup[1]), len(pair_tup[2])) 
                tokenizers = get_tokenizers(sample_dataset) # 删减hyper edge
                hyperg_tup, pair_tup, n_nodes_tup = create_hyperg(train_dataset, tokenizers, ono_level=2) # 前两位相等即可
                # save_pickle(hyperg_tup, root_to + 'hyperg_tup.pkl')
                # save_pickle(pair_tup, root_to + 'pair_tup.pkl')
                # save_pickle(n_nodes_tup, root_to + 'n_nodes_tup.pkl')
                load_dataset = time.time()
                print('Hyper Dataset done!, Cost {} s'.format(load_dataset - endt))

    # STEP 2: load dataloader
    # kg_emb
    # entity_emb = load_pickle(config['KG_DATADIR'] + config['DATASET'] + '/entity_emb.pkl')
    # entity_emb_pad = np.zeros((1, config['KG_DIM']))
    # entity_emb = np.concatenate((entity_emb, entity_emb_pad), axis=0) #加到末尾;这个放到预处理里面


    train_dataloader = get_dataloader(train_dataset, batch_size=config['BATCH'], shuffle=True, drop_last=True) # 得明确一下其是否是经过standarlized
    # val_dataloader = get_dataloader(val_dataset, batch_size=config['BATCH']*5, shuffle=False, drop_last=True)
    test_dataloader = get_dataloader(test_dataset, batch_size=config['BATCH'], shuffle=True, drop_last=True) # config['BATCH']
    load_dataloader = time.time()
    print('Dataloader done!, Cost {} s'.format(load_dataloader - load_dataset))

    gc.collect()

    # STEP 3: define model
    model = HyperRec(
        # basic
        sample_dataset,
        feature_keys=["conditions", "procedures", "drugs"], # drug_his和drug用同一套编码
        label_key="label",
        mode="multiclass",

        # params
        dropout=config['DROPOUT'],
        num_rnn_layers = config['RNN_LAYERS'],
        num_gnn_layers = config['GNN_LAYERS'],
        embedding_dim=config['DIM'],
        hidden_dim=config['DIM'],
        kg_embedding_dim=config['KG_DIM'],
        relation_embedding_dim=config['DIM'],

        # graph related hyper
        pretrained_emb=True,
        n_entity=len(ehr_id_dic)+1, # 留一位给padding; 最后一位
        n_relation=len(rel_dic)+1,

        # hyper g
        hyperg_tup=hyperg_tup,
        pair_tup=pair_tup,
        n_nodes_tup=n_nodes_tup,
    )
    # model = torch.load('/home/czhaobo/HyperHealth/src/output/drugrec-o-MIII/best.ckpt')

    del hyperg_tup, pair_tup, n_nodes_tup # entity_emb
    gc.collect()

    load_model = time.time()
    print('Model done!, Cost {} s'.format(load_model - load_dataloader))
    # del a
    print("Current Using Device", 'cuda:'+config['GPU'])

    # STEP 4: define trainer
    trainer = Trainer(
        model=model,
        # checkpoint_path='/home/czhaobo/KnowHealth/src/output/los-MIII/best.ckpt',
        metrics=["roc_auc_weighted_ovr", "accuracy", "cohen_kappa", "f1_weighted"], # 换指标
        device='cuda:' + config['GPU'],
        exp_name='los-o-' + config['DATASET'],
    )

    trainer.train( # warm cold 这是重新train
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        epochs=config['EPOCH'],
        monitor="roc_auc_weighted_ovr", # roc_auc
        optimizer_params={"lr": config['LR']},
        max_grad_norm=0.1,
        load_best_model_at_last=True
    )
    # trainer.evaluate(test_dataloader)

if __name__ == '__main__':
    print("Hi, This is Hyper Health!")
    run_single_config()
