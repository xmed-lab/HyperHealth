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
    DEV = False # 是否使用sample子集
    MODEL = "HyperRec"
    TASK = 'Los'
    DATASET = 'MIII'
    ATCLEVEL = 3
    RATIO = 0.8 # train-test split
    THRES = 0.4 # pred threshold

    # train parameter
    SEED = 2023
    USE_CUDA = True
    GPU = '7'
    EPOCH = 50
    DIM = 128
    KG_DIM = 768
    LR = 1e-4 # few shot改为1e-3
    BATCH = 4 #
    RNN_LAYERS = 2
    GNN_LAYERS = 2
    DROPOUT = 0.1
    WD = 0

    # loss weight
    MULTI = 0.0
    DDI = 0.0 # target ddi
    DIS =  1e-4  #1e-4
    SSL = 0.000 # global local
    CONS = 0.0
    EDGE = 0.000 # edge mask

    # prepro
    FUZZ=3 # nn fuzz
    MAXSEQ = 10 # mask seq
    MAXCODESEQ = 512
    RANDOM_LENGTH = 5 # 随机采样长度
    RANDOM_NUM = 5 # 随机采样次数; 这样是不是可以构建一个小的KG
    KGPER = 512

    # Hyper
    HYPERG = None
    HYPERATE = 0.001
    SYM_NUM = 16 # num of meta symptom;  
    ITER = 1 # num of iters
    N_FOLD = 1 # num of fold;
    VOTE = 'gate'

    KG_DATADIR = '/home/czhaobo/HyperHealth/data/ready-gpt/'

class HyperLosConfigMIV():
    DEV = False # 是否使用sample子集
    MODEL = "HyperRec"
    TASK = 'Los'
    DATASET = 'MIV'
    ATCLEVEL = 3
    RATIO = 0.8 # train-test split
    THRES = 0.4 # pred threshold

    # train parameter
    SEED = 2023
    USE_CUDA = True
    GPU = '7'
    EPOCH = 20
    DIM = 128
    KG_DIM = 768
    LR = 2e-4
    BATCH = 16 # 应该用32的  感觉大batch会降低PRAUC，不仅来自阈值
    RNN_LAYERS = 2
    GNN_LAYERS = 2
    DROPOUT = 0.3
    WD = 0

    # loss weight
    MULTI = 0.0
    DDI = 0.08 # target ddi
    DIS = 1e-4 # 0.001
    SSL = 0.000 # global local
    CONS = 0
    EDGE = 0.000 # edge mask

    # prepro
    FUZZ=3 # nn fuzz
    MAXSEQ = 10 # mask seq
    MAXCODESEQ = 512
    RANDOM_LENGTH = 5 # 随机采样长度
    RANDOM_NUM = 2 # 随机采样次数; 这样是不是可以构建一个小的KG
    KGPER = 512

    # Hyper
    HYPERG = None
    HYPERATE = 0.001
    SYM_NUM = 16 # num of meta symptom;  （不行换激活函数吧）; embedding层级drop；变为0.3
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

