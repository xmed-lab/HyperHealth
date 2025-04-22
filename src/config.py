# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : config.py
# Time       ：8/3/2024 9:12 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：  > tes.log 2>&1 &
"""




class HyperLosConfig():
    # data_info parameter
    DEV = False
    MODEL = "HyperRec"
    TASK = 'Los'
    DATASET = 'MIII'
    ATCLEVEL = 3
    RATIO = 0.8
    THRES = 0.4

    # train parameter
    SEED = 2023
    USE_CUDA = True
    GPU = '7'
    EPOCH = 50
    DIM = 128
    KG_DIM = 768
    LR = 1e-4
    BATCH = 4 #
    RNN_LAYERS = 2
    GNN_LAYERS = 2
    DROPOUT = 0.1
    WD = 0

    # loss weight
    MULTI = 0.0
    DDI = 0.0
    DIS =  1e-4
    SSL = 0.0001
    CONS = 0.0001
    EDGE = 0.0001

    # prepro
    FUZZ=3
    MAXSEQ = 10
    MAXCODESEQ = 512
    RANDOM_LENGTH = 5
    RANDOM_NUM = 5
    KGPER = 512

    # Hyper
    HYPERG = None
    HYPERATE = 0.001
    SYM_NUM = 16
    ITER = 1
    N_FOLD = 1
    VOTE = 'gate'

    KG_DATADIR = '/home/czhaobo/HyperHealth/data/ready-gpt/'

class HyperLosConfigMIV():
    DEV = False # 是否使用sample子集
    MODEL = "HyperRec"
    TASK = 'Los'
    DATASET = 'MIV'
    ATCLEVEL = 3
    RATIO = 0.8
    THRES = 0.4

    # train parameter
    SEED = 2023
    USE_CUDA = True
    GPU = '7'
    EPOCH = 20
    DIM = 128
    KG_DIM = 768
    LR = 2e-4
    BATCH = 16
    RNN_LAYERS = 2
    GNN_LAYERS = 2
    DROPOUT = 0.3
    WD = 0

    # loss weight
    MULTI = 0.0
    DDI = 0.08 # target ddi
    DIS = 1e-4 # 0.001
    SSL = 0.0001 # global local
    CONS = 0.00001
    EDGE = 0.0001 # edge mask

    # prepro
    FUZZ=3 # nn fuzz
    MAXSEQ = 10 # mask seq
    MAXCODESEQ = 512
    RANDOM_LENGTH = 5
    RANDOM_NUM = 2
    KGPER = 512

    # Hyper
    HYPERG = None
    HYPERATE = 0.001
    SYM_NUM = 16
    ITER = 1 # num of iters
    N_FOLD = 1 # num of fold;
    VOTE='gate'

    KG_DATADIR = '/home/czhaobo/HyperHealth/data/ready-gpt/'



#
config = vars(HyperLosConfig)
config = {k:v for k,v in config.items() if not k.startswith('__')}
config = vars(HyperLosConfigMIV)
config = {k:v for k,v in config.items() if not k.startswith('__')}
 #

