# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : mode_los.py
# Time       ：1/4/2024 8:20 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：
"""

import itertools
import time
import pandas as pd
import torch
import dgl
import os
import math
import pkg_resources
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional, Union
from pyhealth.models.utils import get_last_visit
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.nn.functional import multilabel_margin_loss
from pyhealth.metrics import ddi_rate_score
from pyhealth.models.utils import batch_to_multihot
from pyhealth.models import BaseModel
from pyhealth.medcode import ATC
from pyhealth.tokenizer import Tokenizer
from pyhealth.datasets import SampleEHRDataset
from pyhealth import BASE_CACHE_PATH as CACHE_PATH
from graph import KGEncoder, KGPruning, KGFusion, HyperConv
from utils import pad_g_sequence, load_pickle, off_diagonal
from config import config
from itertools import chain
from loss import loss_focal, loss_asy
from itertools import repeat


class Spatial_Dropout(nn.Module):
    def __init__(self,drop_prob):

        super(Spatial_Dropout,self).__init__()
        self.drop_prob = drop_prob

    def forward(self,inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

    def _make_noise(self,input):
        return input.new().resize_(input.size(0),*repeat(1, input.dim() - 2),input.size(2))


class Dice(nn.Module):
    """
    The Data Adaptive Activation Function in DIN, a generalization of PReLu.
    """

    def __init__(self, emb_size, dim=2, epsilon=1e-8):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        self.alpha = nn.Parameter(torch.zeros((emb_size,))) if self.dim == 2 else nn.Parameter(
            torch.zeros((emb_size, 1)))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out




class Rec_Layer(nn.Module):
    def __init__(self, embedding_dim, voc_size, feature_num=3, vote='max', dropout=0.1):
        super(Rec_Layer,self).__init__()
        self.embedding_dim = embedding_dim
        self.ddi_weight = 0.
        self.multiloss_weight = config['MULTI']
        self.ssl_weight = config['SSL']
        self.cons_weight = config['CONS']
        self.dis_weight = config['DIS']
        self.aux_weight = config['EDGE'] # 边的mask
        self.target_ddi = config['DDI']
        self.vote = vote
        self.feature_num = feature_num

        # sequence and rec layer
        self.rnns_global = torch.nn.TransformerEncoderLayer(
                    d_model=3*embedding_dim, nhead=2, batch_first=True, dropout=0.2)
        self.rnns_local = torch.nn.TransformerEncoderLayer(
                    d_model=3*embedding_dim, nhead=2, batch_first=True, dropout=0.2)
        # self.rnns_global = torch.nn.GRU(3*embedding_dim,3* embedding_dim, num_layers=1, batch_first=True, dropout=dropout)
        # self.rnns_local = torch.nn.GRU(3*embedding_dim,3* embedding_dim, num_layers=1, batch_first=True, dropout=dropout)
        #
        # self.rnns_global = nn.MultiheadAttention(3 * embedding_dim, num_heads=1, batch_first=True)
        # self.rnns_local = nn.MultiheadAttention(3 * embedding_dim, num_heads=1, batch_first=True)

        self.global_expert1 = nn.Sequential(nn.Linear((3+1)*embedding_dim, voc_size, bias=False),
                                            nn.LayerNorm(voc_size),
                                            nn.LeakyReLU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(voc_size, voc_size, bias=False),
                                            # nn.BatchNorm1d(voc_size, affine=False),
                                            )

        self.global_expert = nn.Sequential(nn.LayerNorm(embedding_dim* 4), nn.Linear((3+1)*embedding_dim, voc_size, bias=False))
        self.local_expert = nn.Sequential(nn.LayerNorm(embedding_dim* 4), nn.Linear((3+1)*embedding_dim, voc_size, bias=False))

        self.local_expert1 = nn.Sequential(nn.Linear((3+1)*embedding_dim, voc_size, bias=False),
                                            nn.LayerNorm(voc_size),
                                            nn.LeakyReLU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(voc_size, voc_size, bias=False),
                                            # nn.BatchNorm1d(voc_size, affine=False),
                                            )

        self.decision = nn.Sequential(nn.Linear(8*embedding_dim, embedding_dim//2, bias=False),
                                        nn.LeakyReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(embedding_dim//2, 2, bias=False),
                                        nn.Softmax(dim=1))
        self.dropout_seq = nn.Dropout(dropout)
        self.proj1, self.proj2 = nn.Linear(3 * embedding_dim, embedding_dim, bias=False), nn.Linear(3 * embedding_dim, embedding_dim, bias=False)

        self.mlp = nn.Linear(embedding_dim//config['SYM_NUM'], config['SYM_NUM'],  bias=False)

    def forward(self,
            patient_id,
            patient_emb_global: torch.Tensor,
            patient_emb_local: torch.Tensor,
            drugs: torch.Tensor,
            ddi_adj: torch.Tensor,
            mask: torch.Tensor,
            drug_indexes: Optional[torch.Tensor] = None,
            drug_fea: Optional[tuple] = None,
            aux_reg: Optional[torch.Tensor] = None,
            ):
        emb_split = torch.cat(patient_emb_global[mask].split(self.embedding_dim, dim=1), dim=0) # B,V,D

        global_emb = self.rnns_global(patient_emb_global, src_key_padding_mask=~mask)
        local_emb = self.rnns_global(patient_emb_local, src_key_padding_mask=~mask) # douyon给
        # global_emb, _ = self.rnns_global(patient_emb_global)
        # local_emb, _ = self.rnns_local(patient_emb_local)
        # global_emb, _ = self.rnns_global(patient_emb_global, patient_emb_global, patient_emb_global,
        #                                  key_padding_mask=~mask)  # B, V, 3D
        # local_emb, _ = self.rnns_local(patient_emb_local, patient_emb_local, patient_emb_local,
        #                                key_padding_mask=~mask)  # B, V, 3D

        global_emb = get_last_visit(global_emb, mask) # B, 3D
        local_emb = get_last_visit(local_emb, mask) # B, 3D

        contra_view = (global_emb, local_emb)

        # patient id
        patient_id_global, patient_id_local = self.proj1(patient_id), self.proj2(patient_id) # B,D

        # print("patient id", patient_id)
        # print("global emb", global_emb)
        # print("local emb", local_emb)

        global_emb = torch.cat([patient_id_global, global_emb], dim=1)
        local_emb = torch.cat([patient_id_local, local_emb], dim=1)

        logits_global = self.global_expert(global_emb) # B,V
        logits_global2 = self.global_expert1(global_emb) # B,V
        logits_local = self.local_expert(local_emb)
        logits_local2 = self.local_expert1(local_emb)
        # logits_global, logits_global = torch.max(logits_global, logits_global2), torch.max(logits_local, logits_local2)
        patient_id = torch.cat([global_emb, local_emb], dim=1) # B,2D
        if self.vote =='max':
            logits = torch.max(logits_global, logits_local)
        elif self.vote == 'mean':
            logits = logits_global + logits_local #  + logits_global2 + logits_local2
        elif self.vote == 'gate':
            gate = self.decision(patient_id).unsqueeze(dim=1) # B,1,2
            logits = torch.stack((logits_global, logits_local), dim=1) # B,2,V
            logits = torch.bmm(gate, logits).squeeze(1)

        # print(" gate", gate)


        labels_emb, label_mask = drug_fea # B, D
        # fea_sim = torch.einsum('ik,jk->ij', labels_emb, labels_emb) # M,M

        y_prob = F.softmax(logits, dim=-1)

        label_sim = torch.einsum('ik,jk->ij', y_prob.T, y_prob.T) # M,M
        match_sim = (None, label_sim)

        loss = self.calc_loss(logits, y_prob, ddi_adj, drugs, drug_indexes, match_sim, contra_view, emb_split, aux_reg)
        return loss, y_prob, F.softmax(logits_global,dim=-1), F.softmax(logits_local,dim=-1) #, global_emb+local_emb

    # def ssl_loss_f(self, global_view, local_view, temp=0.1): # fea ssl
    #     T = temp
    #     view_global = F.normalize(global_view, dim=1)
    #     view_local = F.normalize(local_view, dim=1)
    #
    #     view_global_abs = view_global.norm(dim=1)
    #     view_local_abs = view_local.norm(dim=1)
    #
    #     sim_matrix = torch.einsum('ik,jk->ij', global_view, local_view) / torch.einsum('i,j->ij', view_global_abs,
    #                                                                                 view_local_abs)
    #     sim_matrix = torch.exp(sim_matrix / T)
    #     pos_sim = sim_matrix[np.arange(view_global.shape[0]), np.arange(view_local.shape[0])]
    #     loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-8)
    #     loss = -torch.log(loss) # 需要额外取mean
    #     return loss.mean()

    # def ssl_loss_f(self,global_view, local_view, temp=0.005):
    #
    #     # empirical cross-correlation matrix
    #     c = self.bn(global_view).T @ self.bn(local_view)
    #
    #     # sum the cross-correlation matrix between all gpus
    #     c.div_(global_view.shape[0])
    #
    #     on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    #     off_diag = off_diagonal(c).pow_(2).sum()
    #     loss = on_diag + temp * off_diag
    #     return loss

    def ssl_loss(self, global_view, local_view, temp=0.1):
        """local batch loss"""
        global_view = F.normalize(global_view, p=2, dim=-1)
        local_view = F.normalize(local_view,  p=2, dim=-1) # B,d
        pos_score = torch.mul(global_view, local_view).sum(dim=1) # B
        neg_score = torch.matmul(global_view, local_view.transpose(0, 1)) # B,D
        pos_score = torch.exp(pos_score / temp)
        neg_score = torch.exp(neg_score / temp).sum(dim=1)
        loss = -torch.log(pos_score / (neg_score + 1e-8))
        return loss.sum()


    def intent_dis_loss(self, emb, k):
        """k是分成多少意图"""
        batch, dim = emb.shape
        emb = emb.view(batch, k, dim//k)
        emb = emb.view(batch * k, dim//k)
        label = torch.arange(k).repeat(batch).to(emb.device)
        loss = torch.mean(F.cross_entropy(self.mlp(emb), label))
        return loss

    def calc_loss(
        self,
        logits: torch.Tensor,
        y_prob: torch.Tensor,
        ddi_adj: torch.Tensor,
        labels: torch.Tensor,
        label_index: Optional[torch.Tensor] = None,
        match_sim: Optional[torch.Tensor] = None,
        contra_view: Optional[torch.Tensor] = None,
        emb_split: Optional[torch.Tensor] = None,
        aux_reg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        ddi_loss = 0 # (mul_pred_prob * ddi_adj).sum() / (ddi_adj.shape[0] ** 2) # 这里感觉也有问题，是乘以固定值 0.0005

        y_pred = y_prob.detach().cpu().numpy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0 # 优化同步
        y_pred = [np.where(sample == 1)[0] for sample in y_pred]

        loss_cls = F.cross_entropy(logits, labels) # + loss_focal(logits, labels) #+ self.ddi_weight * ddi_loss # 似乎不用
        value_copy = loss_cls.clone().detach().cpu().numpy()

        loss_multi, loss_ssl,loss_ssl,loss_dis, loss_cons = 0,0,0,0,0
        if self.multiloss_weight > 0 and label_index is not None:
            loss_multi = multilabel_margin_loss(y_prob, label_index)
            loss_cls = self.multiloss_weight * loss_multi + (1 - self.multiloss_weight) * loss_cls

        # mask loss
        if self.aux_weight >0 and aux_reg is not None:
            loss_cls = self.aux_weight * aux_reg + loss_cls

        # label cor loss
        if self.cons_weight > 0 and match_sim is not None:
            loss_cons = F.mse_loss(match_sim[0], match_sim[1], reduction='mean')
            loss_cls = self.cons_weight * loss_cons + loss_cls

        # global local contrast
        if self.ssl_weight > 0 and contra_view is not None:
            loss_ssl = self.ssl_loss(contra_view[0], contra_view[1])
            loss_cls = self.ssl_weight * loss_ssl + loss_cls

        # symptom
        if self.dis_weight > 0 and contra_view is not None:
            loss_dis = self.intent_dis_loss(emb_split, config['SYM_NUM'])
            loss_cls = self.dis_weight * loss_dis + loss_cls

        # print("Current loss", loss_dis,  loss_ssl, loss_cons, aux_reg, value_copy) # 2/38/15/0,0.79

        # 作废
        # cur_ddi_rate = ddi_rate_score(y_pred, ddi_adj.cpu().numpy()) # 所有用户的所有ddi
        # if cur_ddi_rate > self.target_ddi:
        #     loss = loss_cls
        # else:
        #     loss = loss_cls

        return loss_cls



class HyperRec(BaseModel):
    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys=["conditions", "procedures", "drugs"],
            label_key="drugs",
            mode="multilabel",

            # hyper related
            dropout: float = 0.3,
            num_rnn_layers: int = 2,
            num_gnn_layers: int = 2,
            embedding_dim: int = 64,
            hidden_dim: int = 64,
            kg_embedding_dim: int = 64,
            relation_embedding_dim: int = 64,

            # graph_related
            pretrained_emb: np.ndarray = None,
            n_entity: int = 100,
            n_relation: int = 1000,
            **kwargs,
    ):
        super(HyperRec, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        # define
        self.num_rnn_layers = num_rnn_layers
        self.num_gnn_layers = num_gnn_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.kg_embedding_dim = kg_embedding_dim # all entity， kg的前k位置最好对齐entity
        self.relation_embedding_dim = relation_embedding_dim

        self.dropout_hp = torch.nn.Dropout(0.3)
        self.dropout_id = torch.nn.Dropout(0.3)

        self.hyperg_tup, self.pair_tup, self.n_nodes_tup = kwargs['hyperg_tup'], kwargs['pair_tup'], kwargs['n_nodes_tup']
        self.pair_tup = [torch.tensor(x, dtype=torch.long, device=self.device)[:, :2].T for x in self.pair_tup] # 暂时不需要typ

        # embedding & tokenizers
        self.kg_node_embedding = torch.nn.Embedding(n_entity, self.kg_embedding_dim, padding_idx=n_entity-1) # 最后一个是pad掉的
        self.kg_edge_embedding = torch.nn.Embedding(n_relation, self.relation_embedding_dim, padding_idx=n_relation-1)
        self.feat_tokenizers = self.get_feature_tokenizers() # tokenizer
        self.label_tokenizer = self.get_label_tokenizer()
        self.label_size = self.label_tokenizer.get_vocabulary_size()

        # save ddi adj
        self.ddi_adj = torch.nn.Parameter(self.generate_ddi_adj(), requires_grad=False)
        ddi_adj = self.generate_ddi_adj() # 用于存储
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj.numpy())

        # module
        self.kg_pruning = KGPruning(self.embedding_dim, self.hidden_dim, self.kg_embedding_dim, self.relation_embedding_dim, dropout=dropout)
        self.kg_encoder = KGEncoder(self.num_gnn_layers-1, self.hidden_dim, self.hidden_dim, self.hidden_dim, output_size=self.embedding_dim, dropout=dropout)
        # from graph import HAR
        # self.kg_encoder = HAR(self.num_gnn_layers, self.hidden_dim, self.hidden_dim, self.hidden_dim, output_size=self.embedding_dim, dropout=dropout)

        self.fusion = KGFusion(embedding_dim, dropout=dropout) # 不一定要fusion

        self.rec_layer = Rec_Layer(self.embedding_dim, self.label_size, dropout=dropout, vote=config['VOTE'])
        self.proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False) # 用于投影

        # 特殊/tmp定义， 不能放在init_weights之前，会使得embedding机制失效
        # init params
        # self.init_weights()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)  # ehr emb
        self.drug_size = self.feat_tokenizers['drugs'].get_vocabulary_size()

        if pretrained_emb:
            pretrained_embs = load_pickle(config['KG_DATADIR'] + config['DATASET'] + '/entity_emb.pkl')
            pretrained_embs = np.concatenate((pretrained_embs, np.zeros((1, config['KG_DIM']))),axis=0) # 加上pad
            self.kg_node_embedding.weight.data.copy_(torch.from_numpy(pretrained_embs))
            self.kg_node_embedding.weight.requires_grad = False

        self.hgats = torch.nn.ModuleDict(
            {
                x: HyperConv(self.hyperg_tup[index], self.embeddings[x].weight, layers=self.num_gnn_layers-1, n_caps=config['SYM_NUM'], routit=config['ITER'], rate=config['HYPERATE'], device=self.device, n_fold=config['N_FOLD'])
                for index, x in enumerate(feature_keys)
            }
        )


    def init_weights(self):
        """Initialize weights. 这个不咋样"""
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)
        self.apply(_init_weights)


    def generate_ddi_adj(self) -> torch.FloatTensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True) # dataframe，这里使用了gamenet的ddi,不要存储
        # ddi = pd.read_csv('/home/czhaobo/KnowHealth/data/drugrec/MIII/processed/ddi_pairs.csv', header=0, index_col=0).values.tolist()
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = np.zeros((self.label_size, self.label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi # each row
        ]

        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        ddi_adj = torch.FloatTensor(ddi_adj)
        return ddi_adj

    def encode_patient(self, feature_key: str, raw_values: List[List[List[str]]]) -> torch.Tensor:
        codes = self.feat_tokenizers[feature_key].batch_encode_3d(raw_values, max_length=[config['MAXSEQ'],config['MAXCODESEQ']]) # 这里会padding, B,V,M
        codes = torch.tensor(codes, dtype=torch.long, device=self.device)
        masks = codes!=0 # B,V,M
        embeddings = self.embeddings[feature_key](codes) # B,V,M,D
        embeddings = self.dropout_id(embeddings)
        visit_emb = torch.sum(embeddings, dim=2) # 
        return codes, embeddings, masks, visit_emb # B,V, D

    # def encode_patient_kg_type(self, kg, max_length=[config['MAXSEQ'],config['KGPER']]): # config['MAXSEQ'],config['MAXCODESEQ']
    #     """可以考虑存储每个用户每次visit的子图，到graph中
    #     设定一个max_length，[max_seq, max_code]
    #     """
    #     kgs = []
    #     for patient in kg:
    #         patient = patient[-max_length[0]:] if max_length else patient
    #         visit_kgs = []
    #         for triple_sets in patient:
    #             triple_sets = np.unique(np.array(triple_sets), axis=0)
    #             triple_sets = triple_sets[-max_length[1]:] if max_length else patient
    #             src_raw, rel_raw, dst_raw = np.transpose(triple_sets)[:, :3] # triple_sets[:, 0:1].transpose()[0], triple_sets[:, 1:2].transpose()[0], triple_sets[:, 2:3].transpose()[0]
    #
    #             unique_node_set, edges = np.unique((src_raw, dst_raw), return_inverse=True)
    #             n_src1, n_dst1 = np.reshape(edges, (2, -1)) # new code，from zero
    #
    #             new_src, new_dst, new_rel = np.concatenate((n_src1, n_dst1)), np.concatenate((n_dst1, n_src1)), np.concatenate((rel_raw, rel_raw)) # 无向
    #             graph = dgl.graph((new_src, new_dst)).to(self.device)
    #
    #             unique_type = torch.from_numpy(new_rel).long().to(self.device)
    #             graph.edata['etype'] = unique_type
    #             graph.edata['fea'] = self.kg_edge_embedding(unique_type)
    #             unique_id = torch.from_numpy(unique_node_set).long().to(self.device) # 记录原始id
    #             graph.ndata['raw_id'] = unique_id
    #             graph.ndata['fea'] = self.kg_node_embedding(unique_id)
    #             visit_kgs.append(graph)
    #         kgs.append(visit_kgs)
    #     return kgs

    def encode_patient_kg_type(self, kg, max_length=[config['MAXSEQ'],config['KGPER']]): # config['MAXSEQ'],config['MAXCODESEQ']
        """可以考虑存储每个用户每次visit的子图，到graph中
        设定一个max_length，[max_seq, max_code]
        """
        kgs, length = [], []
        for patient in kg:
            patient = patient[-max_length[0]:] if max_length else patient
            length.append(len(patient))
            for triple_sets in patient:
                triple_sets = np.unique(np.array(triple_sets), axis=0)
                triple_sets = triple_sets[-max_length[1]:] if max_length else patient
                src_raw, rel_raw, dst_raw = np.transpose(triple_sets)[:, :3] # triple_sets[:, 0:1].transpose()[0], triple_sets[:, 1:2].transpose()[0], triple_sets[:, 2:3].transpose()[0]

                unique_node_set, edges = np.unique((src_raw, dst_raw), return_inverse=True)
                n_src1, n_dst1 = np.reshape(edges, (2, -1)) # new code，from zero

                new_src, new_dst, new_rel = np.concatenate((n_src1, n_dst1)), np.concatenate((n_dst1, n_src1)), np.concatenate((rel_raw, rel_raw)) # 无向
                graph = dgl.graph((new_src, new_dst))

                graph.edata['etype'] = torch.from_numpy(new_rel)
                graph.ndata['raw_id'] = torch.from_numpy(unique_node_set)# 记录原始id
                # visit_kgs.append(graph)
                kgs.append(graph)

        graph_batch = dgl.batch(kgs).to(self.device)
        graph_batch.edata['etype'] = graph_batch.edata['etype'].long()
        graph_batch.ndata['raw_id'] = graph_batch.ndata['raw_id'].long()
        graph_batch.edata['fea'] = self.kg_edge_embedding(graph_batch.edata['etype'])
        graph_batch.ndata['fea'] = self.kg_node_embedding(graph_batch.ndata['raw_id'])
        return graph_batch, length

    def encode_patient_kg(self, conditions_kg, procedures_kg, drugs_hist_kg):
        """所有visit对应的kg合集"""
        conditions_kg_batch, length1 = self.encode_patient_kg_type(conditions_kg) # B,K,D， 不一样长
        procedures_kg_batch, length2 = self.encode_patient_kg_type(procedures_kg)
        drugs_hist_kg_batch, length3 = self.encode_patient_kg_type(drugs_hist_kg)
        return conditions_kg_batch, procedures_kg_batch, drugs_hist_kg_batch, [length1, length2, length3]


    def encode_hyper_emb(self):
        """可能需要预先的pretrain"""
        self.paddings = torch.zeros(1, self.embedding_dim, requires_grad=False).to(self.device)

        self.node_emb_cond, edge_emb_cond = self.hgats['conditions'](self.pair_tup[0])
        self.node_emb_proc, edge_emb_proc = self.hgats['procedures'](self.pair_tup[1])
        self.node_emb_drug, edge_emb_drug = self.hgats['drugs'](self.pair_tup[2])
        # 下面这种方式不够优雅
        self.node_emb_cond = torch.cat([self.paddings,self.node_emb_cond], dim=0) # pad, unk])
        self.node_emb_proc = torch.cat([self.paddings,self.node_emb_proc], dim=0) # pad, unk])
        self.node_emb_drug = torch.cat([self.paddings,self.node_emb_drug], dim=0) # pad, unk])
        # print("AAAAAA", self.node_emb_cond[0])
        # self.node_emb_cond[:2], self.node_emb_proc[:2], self.node_emb_drug[:2] = 0,0,0 # pad, unk
        # print('Check shape,', self.node_emb_cond.shape, self.embeddings['conditions'].weight.shape)
        return

    def encode_hyper_emb_batch(self, cond_codes, proc_codes, drug_codes, labels):
        """可能需要预先的pretrain"""
        self.encode_hyper_emb() # forward
        batch_emb_cond = self.node_emb_cond[cond_codes]
        batch_emb_proc = self.node_emb_proc[proc_codes]
        batch_emb_drug = self.node_emb_drug[drug_codes]
        # sum
        batch_emb_cond, batch_emb_proc, batch_emb_drug = torch.sum(self.dropout_hp(batch_emb_cond), dim=2), torch.sum(self.dropout_hp(batch_emb_proc), dim=2), torch.sum(self.dropout_hp(batch_emb_drug), dim=2) # B,V,D
        if labels:
            batch_emb_label = self.node_emb_drug[labels] # Num,D;
        else:
            batch_emb_label = None

        return batch_emb_cond, batch_emb_proc, batch_emb_drug, batch_emb_label

    def decode_label(self, array_prob, tokenizer):
        """给定y概率，label tokenizer，返回所解码出来的code Token"""
        array_prob[array_prob >= config['THRES']] = 1
        array_prob[array_prob < config['THRES']] = 0 # 优化同步
        indices = [np.where(row == 1)[0].tolist() for row in array_prob]
        tokens = tokenizer.batch_decode_2d(indices)
        return tokens

    # def cor_label_submatrix(self, labels_index):
    #     """便于计算cor label的时候计算编码"""
    #     label = self.label_tokenizer.batch_encode_2d(
    #         labels_index, padding=True
    #     )+2 # B, D
    #     mask = label != 2
    #     label = label * mask
    #     return label, mask
    def cor_label_submatrix(self):
        # 前面两位是pad和unk
        return torch.tensor(list(range(2, self.label_size+2)), device=self.device, dtype=torch.long), 0

    def forward(
        self,
        patient_id : List[List[str]],
        conditions: List[List[List[str]]], # 需要和dataset保持一致[名字，因为**data]
        procedures: List[List[List[str]]],
        drugs: List[List[str]],  # label
        drugs_hist: List[List[List[str]]],
        conditions_kg: List[List[List[int]]],
        procedures_kg: List[List[List[int]]],
        drugs_hist_kg: List[List[List[int]]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.
        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor of shape [patient, visit, num_labels]
                    representing the probability of each drug.
                y_true: a tensor of shape [patient, visit, num_labels]
                    representing the ground truth of each drug.
        """
        # # patient id
        # id_index = self.id_tokenizer.convert_tokens_to_indices(patient_id)
        # patient_id_emb = self.id_embeddings(torch.tensor(id_index, dtype=torch.long, device=self.device))

        # prepare labels
        labels = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)

        # candidate_context
        labels_code = torch.tensor(list(range(2, self.drug_size+2)), device=self.device, dtype=torch.long)

        # labels_index = self.label_tokenizer.batch_encode_2d(
        #     drugs, padding=False, truncation=False
        # ) # [[23,32],[1,2,3]]，注意比feature_tokenizer少两位
        # label_code, label_mask = self.cor_label_submatrix() # B,D
        #
        # labels = batch_to_multihot(labels_index, self.label_size) # tensor, B, Label_size;  # convert to multihot

        # index_labels = -np.ones((len(labels), self.label_size), dtype=np.int64)
        # for idx, cont in enumerate(labels_index):
        #     # remove redundant labels
        #     cont = list(set(cont))
        #     index_labels[idx, : len(cont)] = cont # remove padding and unk
        # index_labels = torch.from_numpy(index_labels) # 类似的！【23，38，39】
        #
        # labels = labels.to(self.device) # for bce loss
        # index_labels = index_labels.to(self.device) # for multi label loss

        # patient id
        cond_code,_, condi_mask, condition_vis_emb = self.encode_patient("conditions", conditions) # [B,V,M] [B,V,M,D]; [B,V,M], [B,V,D]
        proc_code,_, proc_mask, procedure_vis_emb = self.encode_patient("procedures", procedures)
        drug_code,_, drug_mask, drug_history_vis_emb = self.encode_patient("drugs", drugs_hist)

        condition_emb, procedure_emb, drugs_history_emb = condition_vis_emb, procedure_vis_emb, drug_history_vis_emb
        mask = torch.sum(condi_mask, dim=-1) !=0 # visit-level mask; 这个更安全，emb相加可能为0
        # patient_id = torch.cat([condition_emb.sum(dim=1), procedure_emb.sum(dim=1)],dim=1)  / 3 / (mask.sum(dim=1).unsqueeze(dim=1)) # B,2D; ,self.dropout_id(drugs_history_emb).sum(dim=1)
        patient_id = torch.cat([get_last_visit(condition_emb, mask), get_last_visit(procedure_emb,mask), get_last_visit(drugs_history_emb,mask)],dim=1) #  / 3 / (mask.sum(dim=1).unsqueeze(dim=1)) # B,2D; ,self.dropout_id(drugs_history_emb).sum(dim=1)

        # hyper emb
        node_emb_cond, node_emb_proc, node_emb_drug, labels_emb = self.encode_hyper_emb_batch(cond_code, proc_code, drug_code, None) # [B,V,M,D], [Num,D]
        patient_emb_global = torch.cat([node_emb_cond, node_emb_proc, node_emb_drug], dim=-1) # 一般就作为query了 drugs_history_emb; node_emb_drug

        # KG extract, pruning, combination; [[g,g,g],[g,g]]
        conditions_kg, procedures_kg, drugs_hist_kg, length_new = self.encode_patient_kg(conditions_kg, procedures_kg, drugs_hist_kg)
        kgs_new, edge_masks_new, reg_loss = self.kg_pruning((conditions_kg, procedures_kg, drugs_hist_kg), (condition_emb, procedure_emb, drugs_history_emb), mask)
        del conditions_kg, procedures_kg, drugs_hist_kg
        conditions_kg_seq, procedures_kg_seq, drugs_hist_kg_seq = self.kg_encoder(kgs_new, edge_masks_new, length_new)
        conditions_kg_seq, procedures_kg_seq, drugs_hist_kg_seq = F.normalize(conditions_kg_seq, p=2, dim=-1), F.normalize(procedures_kg_seq, p=2, dim=-1), F.normalize(drugs_hist_kg_seq, p=2, dim=-1)
        patient_emb_local = torch.cat([conditions_kg_seq + node_emb_cond, procedures_kg_seq + node_emb_proc,drugs_hist_kg_seq + node_emb_drug], dim=-1) #  加入自回归


        # calculate loss
        loss, y_prob, global_prob, local_prob = self.rec_layer( # patient_emb
            patient_id,
            patient_emb_global=patient_emb_global,
            patient_emb_local=patient_emb_local,
            drugs=labels,
            ddi_adj=self.ddi_adj,
            mask=mask,
            drug_indexes=None,
            drug_fea=(None, None),
            aux_reg=sum(reg_loss)/3,
        )


        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": labels,
            "global_prob": global_prob,
            "local_prob": local_prob,
        #     patient: patient_emb
        }



