#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/12 20:00
# @Author  : lhz
# @File    : my_gat_layer.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        # adj: n x n
        #   h: n x f_in
        n = h.size(0)
        # n x f_in
        # print(self.w.shape)
        h_prime = torch.matmul(h, self.w)
        # n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)
        # n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)
        # n_head x n x 1
        attn = attn_src.expand(-1, -1, n) + \
               attn_dst.expand(-1, -1, n).permute(0, 2, 1)
        # n_head x n x n

        attn = self.leaky_relu(attn)
        mask = 1 - adj.unsqueeze(0)
        # 1 x n x n
        attn.data.masked_fill_(mask.bool(), float("-inf"))
        attn = self.softmax(attn)
        # n_head x n x n
        attn = self.dropout(attn)
        # print('')
        output = torch.matmul(attn, h_prime)
        # n_head x n x f_out
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn
