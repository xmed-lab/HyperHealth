# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : graph_ref.py
# Time       ：8/3/2024 9:13 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：实现一下HyperGraph; Graph
"""
import dgl
import math
import torch as th
import numpy as np
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from itertools import chain, islice

from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from functools import reduce
from utils import pad_batch
import dgl.sparse as dglsp
from torch.nn.parameter import Parameter




# Graph Model


# class Reluctant(nn.Module):
#     def __init__(self, kg_emb_size, relation_embed_size, hidden_size):
#         # self.virtual_emb = virtual_emb # K, D
#         super(Reluctant, self).__init__()
#         self.nodes_proj = nn.Sequential(
#             nn.Linear(kg_emb_size, hidden_size),
#             nn.Dropout(0.5)
#         )
#         self.edges_proj = nn.Sequential(
#             nn.Linear(relation_embed_size, relation_embed_size),
#             nn.Dropout(0.5)
#         )
#
#     def forward(self, kg_batch_lis, virtual_emb):
#         flattened_list = list(chain(*kg_batch_lis))
#         lengths = [len(sublist) for sublist in kg_batch_lis]
#         g_batch = dgl.batch(flattened_list)
#         raw_efea = g_batch.edata['fea'] # all edges , emb
#         # sim
#         relation_ = th.mm(raw_efea, virtual_emb.weight.data.t())
#         relation_remap = th.argmax(relation_, dim=1) # tensor edge_num
#         g_batch.edata['etype'] = relation_remap #
#         g_batch.edata['fea'] = virtual_emb(relation_remap)
#
#         #  nodes_proj
#         g_batch.ndata['fea'] = self.nodes_proj(g_batch.ndata['fea'])
#         g_batch.edata['fea'] = self.edges_proj(g_batch.edata['fea'])
#         kgs_lis = dgl.unbatch(g_batch)
#
#         return kgs_lis, lengths



class Edge_Drop_Learner(nn.Module):
    def __init__(self, node_dim, edge_dim=None, mlp_edge_model_dim=None):
        super(Edge_Drop_Learner, self).__init__()
        self.mlp_src = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.LeakyReLU(negative_slope=0.01), # 这里换成leaky会不会好点
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_dst = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_con = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_edge_model_dim, 1)
        )

        self.mlp_edge = nn.Sequential(
            nn.Linear(edge_dim, mlp_edge_model_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, node_emb, graph, temperature=0.5):
        w_src = self.mlp_src(node_emb)
        w_dst = self.mlp_dst(node_emb)
        graph.srcdata.update({'inl': w_src})
        graph.dstdata.update({'inr': w_dst})
        graph.apply_edges(fn.u_add_v('inl', 'inr', 'ine'))
        n_weight = graph.edata.pop('ine')
        weight = n_weight
        relation_emb = graph.edata['fea']
        if relation_emb is not None and self.mlp_edge is not None:
            w_edge = self.mlp_edge(relation_emb)
            graph.edata.update({'ee': w_edge})
            e_weight = graph.edata.pop('ee')
            weight += e_weight # edge
        weight = weight.squeeze()
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * th.rand(weight.size()) + (1 - bias)
        gate_inputs = th.log(eps) - th.log(1 - eps)
        gate_inputs = gate_inputs.to(node_emb.device)
        gate_inputs = (gate_inputs + weight) / temperature # here use gate
        aug_edge_weight = th.sigmoid(gate_inputs).squeeze() #  or we can use gumbel softmax
        edge_drop_out_prob = 1 - aug_edge_weight

        reg = edge_drop_out_prob.mean() # regularization, 用于计算loss, 应该加上reg，逼迫其用更少的边
        aug_edge_weight = aug_edge_weight # (Edges,1,1), 传入到下面的edge weight
        # print(aug_edge_weight.size())
        return reg, aug_edge_weight


class NonRelevant(nn.Module):
    def __init__(self, kg_emb_sz,  kg_rel_sz,  emb_size, hidden_sz, dropout=0.5):
        super().__init__()
        self.nodes_proj = nn.Sequential(
            nn.Linear(kg_emb_sz, hidden_sz),
            nn.Dropout(dropout) # 0.5
        )
        self.edges_proj = nn.Sequential(
            nn.Linear(kg_rel_sz, hidden_sz),
            nn.Dropout(dropout)
        )

        self.transfer_net = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.Dropout(dropout),
        )

        self.guilde_net = nn.Sequential(
            nn.Linear(emb_size + hidden_sz, hidden_sz), # 原本是kg
            nn.Dropout(dropout))

        self.drop = nn.Dropout(dropout)
        self.epsilon = th.FloatTensor([1e-12])
        self.edge_drop_net = Edge_Drop_Learner(hidden_sz, edge_dim=hidden_sz, mlp_edge_model_dim=hidden_sz) # 给他搞到相同大小

    def mask_attribute(self, concat_emb):
        attri_prob = self.guilde_net(concat_emb) # node_num , 2D
        attri_mask = F.gumbel_softmax(attri_prob, tau=1, hard=False)
        return attri_mask

    def forward(self, g_batch, embs, mask, reluc=False, virtual_emb=None):
        """
        :param kg_batch_lis: graph list
        :param lengths: list
        :param embs: B, T, D
        :param mask: ehr mask

        :return:
        """
        if reluc:
            assert virtual_emb is not None, "Virtual embedding must be provided for reluctant learning."
            raw_efea = g_batch.edata['fea']  # all edges , emb
            relation_ = th.mm(raw_efea, virtual_emb.weight.data.t())
            relation_remap = th.argmax(relation_, dim=1)  # tensor edge_num
            g_batch.edata['etype'] = relation_remap  #
            g_batch.edata['fea'] = virtual_emb(relation_remap)

        embs = embs.detach() # 是否detach
        embs = self.transfer_net(embs[mask]) # M,emb

        # kg_batch_lis = list(chain(*kg_batch_lis))
        # g_batch = dgl.batch(kg_batch_lis)

        g_batch.ndata['fea'] = self.nodes_proj(g_batch.ndata['fea'])
        g_batch.edata['fea'] = self.edges_proj(g_batch.edata['fea'])

        # g_batch.ndata['fea'] = self.drop(g_batch.ndata['fea']) # random mask
        # g_batch.edata['fea'] = self.drop(g_batch.edata['fea']) # random mask

        assert len(embs) == g_batch.batch_size, "Each graph in the batch must have a corresponding embedding."
        cum_nodes = th.cumsum(g_batch.batch_num_nodes(), dim=0)
        # 计算每个子图的节点数
        node_counts = th.cat((cum_nodes[:1], th.diff(cum_nodes)))
        # node_counts = [cum_nodes[0]] + [cum_nodes[i] - cum_nodes[i - 1] for i in range(1, len(cum_nodes))]
        # 直接构造整个嵌入张量
        embeddings_all = th.cat([emb.repeat(count, 1) for emb, count in zip(embs, node_counts)], dim=0)
        g_batch.ndata['fea'] = g_batch.ndata['fea'] * self.mask_attribute(th.cat([g_batch.ndata['fea'], embeddings_all], dim=1))

        # unbatch的写法
        # kg_batch_lis = dgl.unbatch(g_batch)
        # # attribute mask
        # if embs.shape[0] != len(kg_batch_lis):
        #     print("Error shape, ", embs.shape, len(kg_batch_lis))
        # try: # 这里比较耗时0.06s
        #     [kg.apply_nodes(lambda nodes: {'fea': nodes.data['fea'] * self.mask_attribute(th.cat([nodes.data['fea'], emb.repeat(kg.ndata['fea'].shape[0], 1)], dim=1))}) for kg, emb in zip(kg_batch_lis, embs)]
        #     # [kg.apply_nodes(lambda nodes: {'mask': th.cat([nodes.data['fea'], emb.repeat(kg.ndata['fea'].shape[0], 1)], dim=1)}) for kg, emb in
        #     #  zip(kg_batch_lis, embs)]
        #     # attri_mask = [self.mask_attribute(th.cat([kg.ndata['fea'], embs[index].repeat(kg.ndata['fea'].shape[0], 1)], dim=1)) for index, kg in enumerate(kg_batch_lis)]
        # except:
        #     print("Error shape 2, ", embs.shape, len(kg_batch_lis))
        #
        # # attri_mask = th.cat(attri_mask)
        # # g_batch.ndata['fea'] = g_batch.ndata['fea'] * attri_mask # target mask
        # g_batch = dgl.batch(kg_batch_lis)
        # # g_batch.ndata['fea'] = g_batch.ndata['fea'] * self.mask_attribute(g_batch.ndata['mask']) # target mask

        # edge mask
        self.epsilon = self.epsilon.to(g_batch.device)

        # 原本就在这里los
        g_batch.ndata['fea'] = self.drop(g_batch.ndata['fea']) # random mask
        g_batch.edata['fea'] = self.drop(g_batch.edata['fea']) # random mask

        tmp = g_batch.ndata['fea'] / (th.max(th.norm(g_batch.ndata['fea'], dim=1, keepdim=True), self.epsilon))
        regs, edge_mask = self.edge_drop_net(tmp, g_batch, temperature=0.7)
        edge_masks = edge_mask.unsqueeze(dim=0) # 1，edge_num



        # kg_batch_lis = dgl.unbatch(g_batch)
        # kg_batch_lis = [g_batch]

        # regs, edge_masks = 0, []
        # for g in kg_batch_lis:
        #     # h = g.ndata['fea']
        #     # tmp = (h / (th.max(th.norm(h, dim=1, keepdim=True), self.epsilon))) # norm
        #     reg, edge_mask = self.edge_drop_net(g.ndata['fea'], g, temperature=0.7)
        #     regs += reg
        #     edge_mask = edge_mask.unsqueeze(dim=0) # 1，edge_num
        #     edge_masks.append(edge_mask)
        # regs = regs / len(kg_batch_lis)

        return edge_masks, g_batch, regs


class KGPruning(nn.Module):
    def __init__(self, emb_size, hidden_size, kg_emb_size, kg_edge_emb_size, dropout): # 这里的emb_size是hidden dim
        super().__init__()
        self.non_relevant_remove = NonRelevant(kg_emb_size, kg_edge_emb_size, emb_size, hidden_size, dropout=0.5) # 0.5

    def forward(self, kgs, embs, mask,  reluc=False, virtual_emb=None):
        kgs_new, edge_masks_new, regs = [], [], []
        for index, kg_batch_lis in enumerate(kgs):
            edge_masks, kg_batch_lis, reg = self.non_relevant_remove(kg_batch_lis, embs[index], mask,  reluc, virtual_emb)
            kgs_new.append(kg_batch_lis) # [[g,g],]: condition,procedure,drug
            edge_masks_new.append(edge_masks) # [lis[[1*edge,num]],]

            # kgs_tmp = dgl.unbatch(kg_batch_lis)[0] # 注意让batch为1
            # torch.save({'edge_mask': edge_masks}, '/home/czhaobo/HyperHealth/src/edge_mask.pt')
            # import random
            # num = random.randint(0,9)
            # th.save({'node':kg_batch_lis.ndata['raw_id'], 'edge':kg_batch_lis.edges(), 'edge_type':kg_batch_lis.edata['etype'], "edge_mask":edge_masks}, '/home/czhaobo/HyperHealth/src/kg-{}-{}.pt'.format(num,index))

        return kgs_new, edge_masks_new, regs

    # def forward(self, kgs, embs, mask,  reluc=False, virtual_emb=None):
    #     """ablation"""
    #     kgs_new, edge_masks_new, regs = [], [], []
    #     for index, kg_batch_lis in enumerate(kgs):
    #         edge_masks, _, reg = self.non_relevant_remove(kg_batch_lis, embs[index], mask,  reluc, virtual_emb)
    #         kgs_new.append(kg_batch_lis) # [[g,g],]: condition,procedure,drug
    #         edge_masks_new.append(edge_masks) # [lis[[1*edge,num]],]
    #     return kgs_new, [None,None,None], regs

    # def forward(self, kgs, embs, mask):
    #     """"这个用于baseline，对应修改KG的embedding大小"""
    #     kgs_new, edge_masks_new, regs = [], [], []
    #     for index, kg_batch_lis in enumerate(kgs):
    #         kgs_new.append(kg_batch_lis) # [[g,g],]: condition,procedure,drug
    #         edge_masks_new.append(None) # [lis[[1*edge,num]],]
    #     return kgs_new, edge_masks_new, regs



class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        edge_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.7, # edge drop 0.7; los red 0.7
        negative_slope=0.2,
        linear=True,
        activation=F.relu,
        allow_zero_in_degree=False,
        use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        #self.fc_edge=nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_edge = nn.Linear(edge_feats, out_feats * num_heads, bias=False)


        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_edge=nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))


        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop) # 这里改为0.5 这里如果不drop会有问题，nan
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if linear:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)  # resnet
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")            # match the var of relu activation function
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_edge, gain=gain)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_mask_prob=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            h_edge = self.feat_drop(graph.edata['fea'])

            if not hasattr(self, "fc_src"):
                self.fc_src, self.fc_dst = self.fc, self.fc
            # feat_src, feat_dst,feat_edge = h_src, h_dst, h_edge
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            feat_edge = self.fc_edge(h_edge).view(-1, self._num_heads, self._out_feats)

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.dstdata.update({"er": er})
            graph.apply_edges(fn.u_add_v("el", "er", "e")) # src,dst fea==> edge fea

            ee = (feat_edge * self.attn_edge).sum(dim=-1).unsqueeze(-1)
            # ee = feat_edge * self.attn_edge
            graph.edata.update({"e": graph.edata["e"] + ee}) # W*src +W*dst + W*edge

            e = self.leaky_relu(graph.edata["e"])

            if self.training and self.edge_drop > 0:
                perm = th.randperm(graph.number_of_edges(), device=graph.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                a = th.zeros_like(e)
                a[eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
                graph.edata.update({"a": a})
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e)) # Edge_num,这里是根据node的incoming edge计算，然后再重排的

            if edge_mask_prob is not None:
                edge_mask_prob = edge_mask_prob.view(-1, 1).unsqueeze(dim=1)
                graph.edata['a'] = graph.edata['a'] * edge_mask_prob # mask weight

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            degs = graph.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -1)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm

            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
            rst = self._activation(rst)

            graph.ndata['new'] = rst # 所有节点
            # graph_rst = dgl.mean_nodes(graph, feat='new') # 这里sum
            graph_rst = dgl.sum_nodes(graph, feat='new') # 这里sum
            return graph_rst, rst # graph-level, node_level

class myRGAT(nn.Module):
    def __init__(self, num_kg_layers, in_feats, edge_feas, hid_feats, out_feats, num_heads, dropout=0.1):
        """保证in_feats=edge_feas"""
        super(myRGAT, self).__init__()
        self.num_kg_layers = num_kg_layers # 暂时不用
        self.kg_layers = nn.ModuleList()
        # input
        self.kg_layers.append(GATConv(in_feats, hid_feats, edge_feas, num_heads=num_heads))
        # hidden (out层不需要分类)
        for i in range(self.num_kg_layers-2):
            self.kg_layers.append(GATConv(hid_feats*num_heads, hid_feats, edge_feas, num_heads=num_heads))
        if self.num_kg_layers > 1:
            self.kg_layers.append(GATConv(hid_feats*num_heads, out_feats, edge_feas, num_heads=num_heads))
        # self.conv1 = GATConv(in_feats, hid_feats, edge_feas, num_heads=num_heads)
        # self.conv2 = GATConv(hid_feats*num_heads, hid_feats, edge_feas, num_heads=num_heads) # edge在第一次会被放缩到hid_feats
        # self.conv3 = GATConv(in_feats*num_heads, out_feats, in_feats, num_heads=num_heads)
        self.proj = nn.Sequential(
            nn.Linear((self.num_kg_layers-2)*hid_feats*num_heads  + (hid_feats + out_feats) * num_heads, out_feats),
            # nn.Dropout(0.3), # +in_feats
            # nn.Tanh(),
        )
        # self.proj = nn.Linear(hid_feats*num_heads, out_feats,  bias=False)

        print("A", (self.num_kg_layers-2)*hid_feats*num_heads  + (hid_feats + out_feats) * num_heads)
        # self.proj = nn.Linear(hid_feats, out_feats,  bias=False)

    def forward(self, graph, inputs, edge_mask_prob):
        embs = []
        init_emb = dgl.mean_nodes(graph, feat='fea') # Num_g, D
        # embs.append(init_emb)
        nodes_h = inputs # num_nodes,fea
        for i in range(self.num_kg_layers):
            graph_rst, nodes_h = self.kg_layers[i](graph, (nodes_h, nodes_h), edge_mask_prob) # Num_g,D; Num_n,D
            # nodes_h = F.leaky_relu(nodes_h) # 这个不需要了，里面已经高过了
            nodes_h = nodes_h.flatten(1)
            graph_rst = graph_rst.flatten(1) # Num,emb
            # graph_rst = F.normalize(graph_rst, dim=1)
            embs.append(graph_rst)

        embs = self.proj(th.cat(embs, dim=1)) # B,T,3D->B,T,D
        # embs = th.stack(embs, dim=1).sum(dim=1)/self.num_kg_layers # B,3,T,2D->B,T,2D
        # embs = self.proj(embs)
        # print("CCCCC", embs.shape)

        return embs


class KGEncoder(nn.Module):
    def __init__(self, num_kg_layers, input, edge_fea, num_hidden, output_size, heads=2, dropout=0.3): # 前面加了edge attention; 后面家了head
        super(KGEncoder, self).__init__()
        # initial graph
        self.kg_gat = myRGAT(num_kg_layers, input, edge_fea, num_hidden, output_size, heads, dropout)
        self.type_emb = nn.Embedding(3, num_hidden)

    def forward(self, kgs_new, edge_masks_new, length_new):
        embs_new= []
        i=0
        for g_batch, edge_mask, length in zip(kgs_new, edge_masks_new, length_new):
            # encode
            # g_batch = dgl.batch(kg_lis)
            init_embs = g_batch.ndata['fea'] # Num, D
            init_embs =init_embs + self.type_emb.weight[i]
            i+=1
            # edge_mask = th.cat(edge_mask, dim=1) # [1, edge_num]
            embs_lis = self.kg_gat(g_batch, init_embs, edge_mask)
            embs = pad_batch(embs_lis, length) # [[embedd, embed, emb],[embed, embed, 0]]
            embs_new.append(embs) # 这一块估计要看下
        return embs_new


# Hypergraph model
class NeibRoutLayer(nn.Module):
    def __init__(self, num_caps, niter, tau=1.0, n_fold=20):
        super(NeibRoutLayer, self).__init__()
        # 通道数量
        self.k = num_caps
        # routing次数
        self.niter = niter
        self.tau = tau
        self.n_fold = n_fold

    def forward(self, x, adjacency, edge_node, alpha=1.0, flag=True):
        # edge_node = th.LongTensor(edge_node)
        m_all, edge_all_edgesort, node_all_edgesort = edge_node.shape[1], edge_node[0], edge_node[1]  # edge_index, node_index
        node_all_edgesort, edge_all_edgesort = node_all_edgesort.to(x.device), edge_all_edgesort.to(x.device)
        node_all_nodesort, indices_nodesort = node_all_edgesort.sort()  # 对节点进行排序。
        edge_all_nodesort = edge_all_edgesort[indices_nodesort] # 每个节点对应的超边

        n, d = x.shape

        e = adjacency.shape[0]
        k, delta_d = self.k, d // self.k  # embedding分成num_cap份
        x = F.normalize(x.view(n, k, delta_d), dim=2)

        # th.save({'symptom': x.cpu()},'/home/czhaobo/HyperHealth/src/sym.pt')
        x = x.view(n, d)

        # 边表征初始化:e*d
        # print("AAAAAAA", adjacency.shape, x.shape) # 这里padding要从unk开始,不要pad
        edge_embeddings = th.matmul(adjacency, x) # 聚合；[198412，6664] * [6664, D] = [198412, D]， 这个太离谱了。不能加
        edge_embeddings = F.normalize(edge_embeddings.view(e, k, delta_d), dim=2).view(e, d)  # 为啥又重弄回来了。

        # end = time.time()
        # print("Check 1", end-start) # 0.0013
        # fold_len:每个fold的长度
        fold_len = m_all // self.n_fold  # 分块处理，加速计算, 避免爆显存

        # edge routing
        u_edge_all = edge_embeddings

        for clus_iter in range(self.niter): # 迭代次数，迭代过多会oom
            for i_fold in range(self.n_fold):  # 为了处理大规模图必须要分块处理。
                start = i_fold * fold_len
                if i_fold == self.n_fold - 1:
                    end = m_all
                else:
                    end = (i_fold + 1) * fold_len

                edge = edge_all_edgesort[start: end]  # 部分超边 [5,6,77,6] 分批次对节点和边进行处理，免得出问题。
                node = node_all_edgesort[start: end]  # 部分节点 [1,1,1,1]
                m = len(edge) # rate = 0.1; n_fold=200时候为85/230
                # print('One fold Len', m)

                scatter_edge_idx = edge.view(m, 1).expand(m, d)  # E * D [[5,5,5,5], [6,6,6,6]]

                z_node = x[node].view(m, k, delta_d)  # N, k, D
                u_edge = u_edge_all  # E-all , D
                # 下面其实就是一个attention的计算。
                p = (z_node * u_edge[edge].view(m, k, delta_d)).sum(dim=2)  # N,k
                p = F.softmax(p / self.tau, dim=1)  # attention

                # scatter_node:汇聚到边上的节点
                scatter_node = (z_node * p.view(m, k, 1)).view(m, d)  # N.D
                del p
                u_edge = th.zeros(e, d, device=x.device) # E, dim [19w条边，太密集了]; 关键的问题在于为什么是全量的edge
                u_edge.scatter_add_(0, scatter_edge_idx,
                                    scatter_node)  # 这样感觉有问题 啊； 将每个节点（scatter_node）每个dim对于边的贡献累加到对应的边（scatter_edge_idx）上。(这里是把整个dim加上去)；害，其实就是sum pooling
                # u_edge = F.normalize(u_edge.view(e, k, delta_d), dim=2).view(e, d)
                del scatter_node
                u_edge_all += u_edge  # 迭代更新

            u_edge_all = F.normalize(u_edge_all.view(e, k, delta_d), dim=2).view(e, d)  # 每个k进行正则化，对每一个都进行迭代。
        # end2 = time.time()
        # print("Check 2", end2-end)
        # node routing
        u_node_all = x

        for clus_iter in range(self.niter):
            for i_fold in range(self.n_fold):
                start = i_fold * fold_len
                if i_fold == self.n_fold - 1:
                    end = m_all
                else:
                    end = (i_fold + 1) * fold_len

                edge = edge_all_nodesort[start: end] # [1,2,2,1]
                node = node_all_nodesort[start: end] # [1,2,3,4]

                m = len(edge)

                scatter_node_idx = node.view(m, 1).expand(m, d)  # Node diff par

                z_edge = u_edge_all[edge].view(m, k, delta_d)  # E, k, delta-d
                u_node = u_node_all # N,D

                # 这里是attention计算
                p = (z_edge * u_node[node].view(m, k, delta_d)).sum(dim=2)
                p = F.softmax(p / self.tau, dim=1)
                scatter_edge = (z_edge * p.view(m, k, 1)).view(m, d) # E,D
                del p
                u_node = th.zeros(n, d, device=x.device)
                u_node.scatter_add_(0, scatter_node_idx, scatter_edge)  # 根据node索引，进行边聚合。
                # 加上节点原本的表征
                u_node_all += u_node
                # end_32 = time.time()
                # print("Check 3.2", end_32-end_31)
            u_node_all = F.normalize(u_node_all.view(n, k, delta_d), dim=2).view(n, d)
        # end3 = time.time()
        # print("Check 3", end3-end2)
        return u_node_all, u_edge_all


class HyperConv(nn.Module):
    def __init__(self, adjacency, embedding, layers, n_caps, routit, device, dim_vertex=64, p_dropout=0.1, rate=1.0,
                 n_fold=10):
        '''
        layers:图卷积层数
        n_caps:解耦通道数
        routit:routing次数
        n_nodes:输入维度
        dim_vertex:表征维度
        '''
        super(HyperConv, self).__init__()
        self.dim_vertex = dim_vertex
        self.layers = layers
        self.n_caps = n_caps
        self.rouit = routit
        self.rate = rate
        self.device = device
        conv_list = []
        if layers == 0:
            self.conv_list = conv_list
        else:
            for i in range(layers):
                conv = NeibRoutLayer(n_caps, routit, n_fold=n_fold)
                conv_list.append(conv)
        self.conv_list = conv_list

        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col)) # edge 超边

        i = th.LongTensor(indices)
        v = th.FloatTensor(values)

        adjacency = th.sparse.FloatTensor(i, v, th.Size(adjacency.shape))
        self.register_buffer('adjacency', adjacency)

        self.p_dropout = p_dropout

        # item表征
        # embedding = nn.Embedding(embedding.shape[0], embedding.shape[-1]) # 这里unk是可以学的，不需要padding
        self.embedding = embedding
        # nn.init.xavier_uniform_(embedding.weight)
        # self.embedding = embedding.weight


        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim_vertex)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _dropout(self, x):
        return F.dropout(x, self.p_dropout, training=self.training)

    def _edge_sampling(self, edge_index, rate=1.0):  # random mask;不是每个节点都能踩到，因为repalce==True
        """如果没踩到就只能用其最开始卷积的edge embedding"""
        n_edges = edge_index.shape[1]  # (E,N), 感觉这里会很耗时
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)

        return edge_index[:, random_indices]


    def forward(self, edge_node):
        if self.layers == 0:
            item_embeddings = self._dropout(self.embedding[1:, :]) # .weight
            item_embeddings = F.normalize(item_embeddings, dim=1)
            edge_embeddings = th.matmul(self.adjacency, item_embeddings)  # 卷积
            edge_embeddings = F.normalize(edge_embeddings, dim=1)

            return item_embeddings, edge_embeddings

        item_embeddings = self._dropout(self.embedding[1:, :]) # 这里需要变化一下 weight

        item_embedding_layer0 = item_embeddings
        final_item = [item_embedding_layer0]  # 最初的节点表征
        final_edge = []

        edge_node_sample = self._edge_sampling(edge_node, self.rate)  # edge-node pair
        # edge_node_sample = edge_node

        # cond = [i for i in edge_node_sample if i[1]==0]

        # 超图卷积
        for conv in self.conv_list:
            item_embeddings, edge_embeddings = conv(item_embeddings, self.adjacency,
                                                    edge_node_sample) # 感觉这里会污染item emb

            item_embeddings = self._dropout(item_embeddings)
            edge_embeddings = self._dropout(edge_embeddings)

            item_embeddings = F.normalize(item_embeddings, dim=1)
            edge_embeddings = F.normalize(edge_embeddings, dim=1)

            final_item.append(item_embeddings)
            final_edge.append(edge_embeddings)


        # 每一层的item表征
        final_item = th.stack(final_item)
        item_embeddings = th.sum(final_item, 0) / (self.layers)

        final_edge = th.stack(final_edge)
        edge_embeddings = th.sum(final_edge, 0) / (self.layers)

        return item_embeddings, edge_embeddings


class HGNN(nn.Module):
    """这里是HGNN, https://github.com/akaxlh/SHT/blob/main/thVersion/Model.py"""
    def __init__(self, H, embeddings):
        super(HGNN, self).__init__()
        self.embedding = embeddings
        self.adjacency = H # 邻接矩阵
        init = nn.init.xavier_uniform_
        self.adjacency_edge_emb = nn.Parameter(init(th.empty(self.adjacency.shape[0], self.embedding.shape[-1])))

    def forward(self, pair_nodes):
        # 可以pair_nodes采样
        embedding = self.embedding
        res = embedding @ (self.adjacency_edge_emb.T @ self.adjacency_edge_emb) # D,dim e
        # print("BBBBBB", res.shape, res.device)
        return res


class HGCN(nn.Module):
    """HGCN, dgl实现"""
    def __init__(self, H, embeddings, hidden_dims=64):
        super().__init__()
        self.X = embeddings
        in_size = out_size = embeddings.shape[1]
        self.W1 = nn.Linear(in_size, hidden_dims)
        self.W2 = nn.Linear(hidden_dims, out_size)
        self.dropout = nn.Dropout(0.5)

        ###########################################################
        # (HIGHLIGHT) Compute the Laplacian with Sparse Matrix API
        ###########################################################
        # H = th.tensor(H.toarray())
        # H = th.nonzero(H).T
        H = th.tensor([H.rows, H.cols])
        H = dglsp.spmatrix(H)
        H = H + dglsp.identity(H.shape)
        # Compute node degree.
        d_V = H.sum(1)
        # Compute edge degree.
        d_E = H.sum(0)
        # Compute the inverse of the square root of the diagonal D_v.
        D_v_invsqrt = dglsp.diag(d_V**-0.5)
        # Compute the inverse of the diagonal D_e.
        D_e_inv = dglsp.diag(d_E**-1)
        # In our example, B is an identity matrix.
        n_edges = d_E.shape[0]
        B = dglsp.identity((n_edges, n_edges))
        # Compute Laplacian from the equation above.
        self.L = D_v_invsqrt @ H @ B @ D_e_inv @ H.T @ D_v_invsqrt

    def forward(self, pair_tup):
        X = self.X
        X = self.L @ self.W1(self.dropout(X))
        X = F.relu(X)
        X = self.L @ self.W2(self.dropout(X))
        return X

#####HTransformer

class HyperGraphAttentionLayerSparse(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, transfer, concat=True, bias=False):
        super(HyperGraphAttentionLayerSparse, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.droplayer = nn.Dropout(self.dropout)

        self.transfer = transfer
        self.edge_node_fusion_layer = nn.Linear(2 * out_features, out_features)

        if self.transfer:
            self.weight = Parameter(th.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = Parameter(th.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(th.Tensor(self.out_features, self.out_features))
        self.linea = nn.Linear(self.in_features, out_features, bias=False)
        # self.weight2 = nn.Linear(self.in_features, self.out_features, bias=False)
        # self.weight3 = nn.Linear(self.out_features, self.out_features, bias=False)

        if bias:
            self.bias = Parameter(th.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.word_context = nn.Embedding(1, self.out_features)

        self.a = nn.Parameter(th.zeros(size=(2 * out_features, 1)))
        self.a2 = nn.Parameter(th.zeros(size=(2 * out_features, 1)))
        # self.a = nn.Linear(2*out_features,1, bias=False)
        # self.a2 = nn.Linear(2*out_features,1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.uniform_(self.a2.data, -stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)
    #

    def forward(self, item_embeddings ,adj):
        """修改最原始的forward"""
        item_embeddings = item_embeddings.squeeze(dim=0)
        # item_embeddings = item_embeddings.matmul(self.weight2)
        # adj = adj.squeeze(dim=0)
        # item_embeddings = F.dropout(x, self.dropout, training=self.training)  # .weight
        # item_embeddings = self.droplayer(x)
        item_embeddings = F.normalize(item_embeddings, dim=-1)
        edge_embeddings = th.matmul(adj, item_embeddings)  # 卷积
        # edge_embeddings = F.normalize(edge_embeddings, dim=-1)

        # attn = th.matmul(edge_embeddings, item_embeddings.T)
        # attn = self.droplayer(attn)
        # attn = F.softmax(attn + 1e-8, dim=-1)

        # zero_vec = -9e15 * th.ones_like(attn) # 如果不构造这个矩阵会不会身下现存
        # attn = th.where(adj.to_dense() > 0, attn, zero_vec) # E*N
        # node2edge = th.matmul(attn, item_embeddings)

        # node2edge = F.dropout(node2edge, self.dropout, training=self.training)
        # node2edge = self.droplayer(node2edge)
        # node2edge1 = F.normalize(node2edge, dim=-1)
        # node2edge = node2edge1.matmul(self.weight3)

        # item_embeddings = item_embeddings
        # attn = node2edge.matmul(item_embeddings.T)# N*E
        # attn = self.droplayer(attn)
        # attn = F.dropout(attn, self.dropout, training=self.training)
        # attn = F.softmax(attn + 1e-8, dim=-1)

        # edge2node = th.matmul(attn.T,node2edge1)
        # edge2node = F.dropout(edge2node, self.dropout, training=self.training)
        # edge2node = self.droplayer(edge2node)
        edge2node = th.matmul(adj.T, edge_embeddings)
        edge2node = F.normalize(edge2node, dim=-1)

        return edge2node


    def forward_x(self, x ,adj):
        """修改最原始的forward"""
        x = x.squeeze(dim=0)
        x = x.matmul(self.weight2)
        # x = F.dropout(x, self.dropout, training=self.training)  # .weight
        item_embeddings = self.droplayer(x)
        item_embeddings = F.normalize(item_embeddings, dim=-1)
        edge_embeddings = th.matmul(adj, item_embeddings)  # 卷积
        edge_embeddings = F.normalize(edge_embeddings, dim=-1)

        attn = th.matmul(edge_embeddings, item_embeddings.T)
        attn = self.droplayer(attn)
        attn = F.softmax(attn + 1e-8, dim=-1)


        node2edge = th.matmul(attn, item_embeddings)

        node2edge = self.droplayer(node2edge)
        node2edge1 = F.normalize(node2edge, dim=-1)
        node2edge = node2edge1.matmul(self.weight3)

        item_embeddings = item_embeddings
        attn = node2edge.matmul(item_embeddings.T)# N*E
        attn = self.droplayer(attn)
        attn = F.softmax(attn + 1e-8, dim=-1)

        edge2node = th.matmul(attn.T,node2edge1)
        edge2node = self.droplayer(edge2node)
        edge2node = F.normalize(edge2node, dim=-1)

        return edge2node



    def forwards(self, x, adj):
        x_4att = x.matmul(self.weight2)

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias

        N1 = adj.shape[1]  # number of edge
        N2 = adj.shape[2]  # number of node

        pair = adj._indices()# 2*16

        get = lambda i: x_4att[i][adj._indices()[1]]
        x1 = th.cat([get(i) for i in th.arange(x.shape[0]).long()]) # 这种方式果断不行啊，这太大了

        print("x1:{}".format(x1.shape)) # 16, 128

        q1 = self.word_context.weight[0:].view(1, -1).repeat(x1.shape[0], 1).view(x1.shape[0], self.out_features)
        print("q1:{}".format(q1.shape)) # 16, 128

        pair_h = th.cat((q1, x1), dim=-1)
        print("pair_h:{}".format(pair_h.shape)) # 16, 256
        pair_e = self.leakyrelu(th.matmul(pair_h, self.a).squeeze()).t()
        print("pair_e:{}".format(pair_e.shape)) # 16
        assert not th.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = th.sparse_coo_tensor(pair, pair_e, th.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15 * th.ones_like(e)
        attention = th.where(adj.to_dense() > 0, e, zero_vec)

        attention_edge = F.softmax(attention, dim=2)
        # print("attention_edge:{}".format(attention_edge[0, 0]))

        edge = th.matmul(attention_edge, x)

        edge = F.dropout(edge, self.dropout, training=self.training)

        edge_4att = edge.matmul(self.weight3)

        get = lambda i: edge_4att[i][adj._indices()[0]]
        y1 = th.cat([get(i) for i in th.arange(x.shape[0]).long()])

        get = lambda i: x_4att[i][adj._indices()[1]]
        q1 = th.cat([get(i) for i in th.arange(x.shape[0]).long()])

        pair_h = th.cat((q1, y1), dim=-1)
        pair_e = self.leakyrelu(th.matmul(pair_h, self.a2).squeeze()).t()
        assert not th.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = th.sparse_coo_tensor(pair, pair_e, th.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15 * th.ones_like(e)
        attention = th.where(adj.to_dense() > 0, e, zero_vec)

        attention_node = F.softmax(attention.transpose(1, 2), dim=2)
        # print("attention_node:{}".format(attention_node[0, 0]))

        edge_feature = th.matmul(attention_node, edge)
        node = th.cat((edge_feature, x), dim=-1)
        node = F.dropout(self.edge_node_fusion_layer(node), self.dropout, training=self.training)

        if self.concat:
            node = F.elu(node)

        return node

    def forwards(self, x, adj):
        x_4att = x.matmul(self.weight2)

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias

        N1 = adj.shape[1]  # number of edge
        N2 = adj.shape[2]  # number of node

        pair = adj.nonzero().t() # 2*16

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        x1 = th.cat([get(i) for i in th.arange(x.shape[0]).long()])
        print("x1:{}".format(x1.shape)) # 16, 128

        q1 = self.word_context.weight[0:].view(1, -1).repeat(x1.shape[0], 1).view(x1.shape[0], self.out_features)
        print("q1:{}".format(q1.shape)) # 16, 128

        pair_h = th.cat((q1, x1), dim=-1)
        print("pair_h:{}".format(pair_h.shape)) # 16, 256
        pair_e = self.leakyrelu(th.matmul(pair_h, self.a).squeeze()).t()
        print("pair_e:{}".format(pair_e.shape)) # 16
        assert not th.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = th.sparse_coo_tensor(pair, pair_e, th.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15 * th.ones_like(e)
        attention = th.where(adj > 0, e, zero_vec)

        attention_edge = F.softmax(attention, dim=2)
        # print("attention_edge:{}".format(attention_edge[0, 0]))

        edge = th.matmul(attention_edge, x)

        edge = F.dropout(edge, self.dropout, training=self.training)

        edge_4att = edge.matmul(self.weight3)

        get = lambda i: edge_4att[i][adj[i].nonzero().t()[0]]
        y1 = th.cat([get(i) for i in th.arange(x.shape[0]).long()])

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        q1 = th.cat([get(i) for i in th.arange(x.shape[0]).long()])

        pair_h = th.cat((q1, y1), dim=-1)
        pair_e = self.leakyrelu(th.matmul(pair_h, self.a2).squeeze()).t()
        assert not th.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = th.sparse_coo_tensor(pair, pair_e, th.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15 * th.ones_like(e)
        attention = th.where(adj > 0, e, zero_vec)

        attention_node = F.softmax(attention.transpose(1, 2), dim=2)
        # print("attention_node:{}".format(attention_node[0, 0]))

        edge_feature = th.matmul(attention_node, edge)
        node = th.cat((edge_feature, x), dim=-1)
        node = F.dropout(self.edge_node_fusion_layer(node), self.dropout, training=self.training)

        if self.concat:
            node = F.elu(node)

        return node

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Attention(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(Attention, self).__init__()
        self.W_Q = nn.Parameter(th.zeros(size=(in_dim, hid_dim)))
        self.W_K = nn.Parameter(th.zeros(size=(in_dim, hid_dim)))
        nn.init.xavier_uniform_(self.W_Q.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_K.data, gain=1.414)

    def forward(self, Q, K, mask=None):

        KW_K = th.matmul(K, self.W_K)
        QW_Q = th.matmul(Q, self.W_Q)
        if len(KW_K.shape) == 3:
            KW_K = KW_K.permute(0, 2, 1)
        elif len(KW_K.shape) == 4:
            KW_K = KW_K.permute(0, 1, 3, 2)
        att_w = th.matmul(QW_Q, KW_K).squeeze(1)

        if mask is not None:
            att_w = th.where(mask == 1, att_w.double(), float(-1e10))

        att_w = F.softmax(att_w / th.sqrt(th.tensor(Q.shape[-1])), dim=-1)
        return att_w

class HGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, drop_out):
        super(HGCN, self).__init__()
        self.edge_att = Attention(in_dim, hid_dim)
        self.node_att = Attention(in_dim, hid_dim)
        self.edge_linear_layer = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, adj):
        edge_fea = th.matmul(adj, x) / (th.sum(adj, dim=-1).unsqueeze(-1) + 1e-5)
        edge_fea = self.dropout(F.leaky_relu(self.edge_linear_layer(edge_fea))) #(batch, edge_num, fea_dim)

        att_e = self.edge_att(edge_fea, x, adj) #(batch, edge_num, node_num)
        edge_fea = th.matmul(att_e.float(), x.float()) #(batch, edge_num, fea_dim)
        att_n = self.node_att(x, edge_fea)
        x = th.matmul(att_n.float(), edge_fea.float())
        return F.leaky_relu(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(max_len + 1, 1, d_model)
        pe[1:, 0, 0::2] = th.sin(position * div_term)
        pe[1:, 0, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, sec_pos_label=None, in_sec_pos_label=None):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        if sec_pos_label is not None:
            x += self.pe[sec_pos_label.long().permute(1, 0), 0, :] * 1e-3
        if in_sec_pos_label is not None:
            x += self.pe[in_sec_pos_label.long().permute(1, 0), 0, :] * 1e-3
        return self.dropout(x).permute(1, 0, 2)


# class HGNN_ATT(nn.Module):
#     def __init__(self, input_size, n_hid, output_size, dropout=0.3):
#         super(HGNN_ATT, self).__init__()
#         self.dropout = dropout
#         self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False,
#                                                    concat=True)
#         self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=False,
#                                                    concat=True)
#
#
#     def forward(self, x, H):
#         x_res = x
#         x = self.gat1(x, H) + x_res
#         x = F.dropout(x, self.dropout, training=self.training)
#         x_res = x
#         x = self.gat2(x, H) + x_res
#         return x


class Hyper_Atten_Block(nn.Module):
    def __init__(self, input_size, head_num, head_hid, dropout):
        super().__init__()
        self.dropout = dropout
        self.gat = nn.ModuleList([HyperGraphAttentionLayerSparse(input_size, head_hid, dropout=self.dropout, alpha=0.2, transfer=True,
                                            concat=True) for _ in range(head_num)])

        self.ln = nn.LayerNorm([input_size])
        self.hm = nn.Sequential(nn.Linear(head_num * head_hid, input_size),
                                 nn.LeakyReLU(),
                                 nn.Dropout(self.dropout),
                                 )

        self.ffn = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            # nn.LayerNorm([input_size]),
            nn.Dropout(self.dropout),
        )

    def forward(self, x, H):
        x_res = x
        x = th.concat([gat(x, H) for gat in self.gat], dim=-1)
        x = self.hm(x) + x_res
        # x = self.ln(x) # 这里不一定要
        x = self.ffn(x) + x
        # x = self.ln(x)
        return x


class HGNN_ATT_MH(nn.Module):
    def __init__(self, input_size, head_num, head_hid, output_size, layers=1, dropout=0.3):
        super(HGNN_ATT_MH, self).__init__()
        self.dropout = dropout
        self.layers = layers
        self.hatt = nn.ModuleList([Hyper_Atten_Block(input_size, head_num, head_hid, dropout) for _ in range(layers)])


    def forward(self, x, H):
        for att in self.hatt:
            x = att(x, H) + x
        return x

class HGraph_Sum(nn.Module):
    """https://github.com/hpzhang94/hegel_sum/blob/main/layers.py"""
    def __init__(self, input_size, n_hid, dropout=0.3, g_layer=1, hidden_size=None, edges=None):
        super(HGraph_Sum, self).__init__()
        self.dropout = dropout
        # self.position = PositionalEncoding(input_size)
        hidden_size = input_size
        self.hgraph_layer = HGNN_ATT_MH(hidden_size, head_num=1, head_hid=hidden_size, output_size=hidden_size, dropout=dropout, layers=g_layer)
        self.ln = nn.LayerNorm([hidden_size])
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, n_hid)
        self.drop_layer = nn.Dropout(self.dropout)
        # self.final_layer = nn.Linear(n_hid, 1)

        values = edges.data
        indices = np.vstack((edges.row, edges.col)) # edge 超边
        i = th.LongTensor(indices)
        v = th.FloatTensor(values)
        edges = th.sparse.FloatTensor(i, v, th.Size(edges.shape))
        self.register_buffer('edges', edges)


    def forward(self, feature, mask=None, sec_pos_label=None, in_sec_pos_label=None, pos_label=None):

        feature = feature[1:, :]

        feature = feature.unsqueeze(0)#, edges.unsqueeze(0) # edges
        # feature = self.position(feature, sec_pos_label, in_sec_pos_label)
        feature = F.leaky_relu(self.input_layer(feature))

        feature = self.hgraph_layer(feature, self.edges)

        feature = F.leaky_relu(self.output_layer(feature))
        feature = self.drop_layer(feature).squeeze(dim=0)
        # feature = self.final_layer(feature)

        # if not self.training:
        #     feature = th.where(mask==1, feature.double(), -1e3) # 使用全量

        return feature


if __name__ == '__main__':
    """这里是测试代码"""
    embedding_size = 128
    output_sz = 64
    model = HGraph_Sum(input_size=embedding_size, n_hid=embedding_size).cuda()
    # x, H = th.randn(4, 128).cuda(), th.randn(4, 4).cuda() # 这里感觉是每个句子搞一个超图，但是这里我们不需要
    # out = model(x, H)
    # print(out.shape)
    print("+"*10)
    import torch
    from scipy.sparse import csr_matrix
    x = th.randn(5,128).cuda()
    rows = np.array([0, 1, 2, 3])
    cols = np.array([1, 2, 0, 3])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    H = csr_matrix((values, (rows, cols)), shape=(4, 4)).tocoo()
    print(H.row)
    out = model(x, H)
    print(out.shape)


    # def matrix_to_pairs(matrix):
    #     pos = th.nonzero(matrix).T
    #     return pos
    #
    # x = th.randn(10,16)
    # H = th.randn(10,10)
    # pos = th.nonzero(matrix).T
    #
    # H = matrix_to_pairs(H)
    # H = dglsp.spmatrix(H)
    # H = H + dglsp.identity(H.shape)
    # model = HGNN(H, in_size=16, out_size=16)
    # out = model(x)
    # print(out.shape)



