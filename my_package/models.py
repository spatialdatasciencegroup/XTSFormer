import math
import pickle
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import math
import torch.nn.functional as F
from xts_package.plot import *
from xts_package.data_preprocess_before import *
from xts_package.data_preprocess_after import *
from xts_package.data_load import *


class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=True)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        # x = self.layer_norm(x)
        # print("mergeLayer dimension", x.shape)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # print("q and k is: ", q, k)
        # torch.bmm: (b,h,w)x(b,w,m)=(b,h,m)
        attn = torch.bmm(q, k.transpose(1, 2))
        # attn: [468669, 1, 5]
        attn = attn / self.temperature

        # print("atten shape", attn.shape)
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        # attn: [468669, 1, 5]
        attn = self.dropout(attn)  # [n * b, l_v, d]

        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head  # 1
        self.d_k = d_k  # 43
        self.d_v = d_v  # 24

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)  # (64, 1*43)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)  # (64, 1*43)
        # self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.w_vs = nn.Linear(d_v, n_head * d_v, bias=False)  # (24, 1*24)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        # self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.attention = ScaledDotProductAttention(temperature=0.1, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)  # (24, 64)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, pre_v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # d_k: 18, d_v: 4, n_head: 1
        # q: [468669, 1, 18]
        # k: [468669, 5, 18]
        # v: [468669, 5, 4]
        # pre_v: [468669, 1, 4]
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        sz_b, len_pre_v, _ = pre_v.size()

        # residual = q
        # q = q.view(sz_b, len_q, 1, d_k)
        # # q: torch.Size([468669, 1, 1, 18])
        # k = k.view(sz_b, len_k, 1, d_k)
        # # k: torch.Size([468669, 5, 1, 18])
        # v = v.view(sz_b, len_v, 1, d_v)
        # # v: torch.Size([468669, 5, 1, 4])

        # q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        # # q: [468669, 1, 18]
        # k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        # # k: [468669, 5, 18]
        # v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        # # v: [468669, 5, 4]

        output, attn = self.attention(q, k, v, mask=mask)
        # output: [468669, 1, 18]
        # attn: [468669, 1, 5]
        # output = output.view(-1, sz_b, len_q, d_v)
        # output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output)
        # output: [468669, 1, 32]

        return output, attn


class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """

    def __init__(self, feat_dim, space_time_dim, n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.file_name = "data/debug/attn.txt"
        self.feat_dim = feat_dim  # 4
        self.space_time_dim = space_time_dim

        self.model_dim = self.feat_dim + self.space_time_dim  # 4+28
        # self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        # self.act = torch.nn.ReLU()

        assert (self.model_dim % n_head == 0)
        # self.logger = logging.getLogger(__name__)

        self.multi_head_target = MultiHeadAttention(n_head,  # 1
                                                    d_model=self.model_dim,  # 24+40=64
                                                    d_k=self.feat_dim + int(self.space_time_dim / 2),
                                                    # self.model_dim, #self.model_dim // n_head,
                                                    d_v=self.feat_dim + int(self.space_time_dim / 2),
                                                    # 24 self.model_dim, #self.model_dim // n_head,
                                                    dropout=drop_out)

    def forward(self, src, src_t, seq, seq_t):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.
        returns:
          output, weight
          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """
        # src: [468669, 4]
        # src_t: [468669, 1, 14]
        # seq: [468669, 5, 4]
        # seq_t: [468669, 5, 14]
        src = torch.unsqueeze(src, dim=1)  # src [B, 1, D]
        # src: [468669, 1, 4]
        src_e_ph = torch.zeros_like(src)
        # src: torch.Size([128, 1, 24])

        # feat_dim = src.size()[-1]
        # src_ext = src[:, :, :feat_dim - 1]
        # # src_ext: torch.Size([468669, 1, 3])
        # seq_ext = seq[:, :, :feat_dim - 1]
        # # seq_ext: torch.Size([468669, 5, 3])

        src_ext = src
        seq_ext = seq
        lambda_weight = 1.0
        # src_t: [468669, 1, 14]
        q = torch.cat([src_ext, lambda_weight * src_t], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]
        # q: [468669, 1, 18]
        k = torch.cat([seq_ext, lambda_weight * seq_t], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]
        # k: [468669, 5, 18]
        output, attn = self.multi_head_target(q=q, k=k, v=k, pre_v=src, mask=None)
        # output: [B, 1, D + Dt], attn: [B, 1, N]
        # output: [468669, 1, 32]
        # attn: [468669, 1, 5]
        output = output.squeeze()
        # output: [468669, 32]
        attn = attn.squeeze()
        # attn: [468669, 5]

        return output, attn


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()

        self.time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** (np.linspace(0, 2, self.time_dim) - 1))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())
        # self.semantic_weights = nn.Parameter()
        # self.weight = nn.Linear(self.nontime_dim, self.)
        # self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        # torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        # ts: [N, L]
        # ts: [468669, 1]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        # ts: [468669, 1, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts = map_ts + self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)

        return harmonic

class Mercer_TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, nontime_feat_dim):
        super(Mercer_TimeEncode, self).__init__()

        self.time_dim = expand_dim
        self.nontime_feat_dim = nontime_feat_dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** (np.linspace(0, 2, self.time_dim) - 1))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())

        self.init_frequency = torch.nn.Parameter(torch.zeros(self.nontime_feat_dim).float())

        self.weights = nn.Linear(self.nontime_feat_dim, self.time_dim)
        # self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        # torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts, nontime_feat):
        # ts: [N, L]
        # ts: [128000, 5]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        # ts: [128000, 5, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts = map_ts + self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        # harmonic: [128000, 5, 14]
        self.my_param = nn.Parameter(torch.randn(nontime_feat.shape))
        weights = self.weights(self.my_param)

        weights = F.softmax(weights, dim=2)
        harmonic = torch.mul(weights, harmonic)

        return harmonic

class FCTimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, nontime_feat_dim):
        super(FCTimeEncode, self).__init__()

        self.time_dim = expand_dim
        self.nontime_feat_dim = nontime_feat_dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** (np.linspace(0, 2, self.time_dim) - 1))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())
        self.weights = nn.Linear(self.nontime_feat_dim, self.time_dim)
        # self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        # torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts, nontime_feat):
        # ts: [N, L]
        # ts: [128000, 5]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        # ts: [128000, 5, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts = map_ts + self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        # harmonic: [128000, 5, 14]
        weights = self.weights(nontime_feat)
        weights = F.softmax(weights, dim=2)
        harmonic = torch.mul(weights, harmonic)

        return harmonic



class Weibull_Decoder(nn.Module):
    def __init__(self, embedding_all_dim, out_layers):
        super(Weibull_Decoder, self).__init__()
        self.fc = nn.Linear(embedding_all_dim, out_layers)

    def forward(self, x):
        y = self.fc(x)
        y = torch.exp(y)
        return y


def task_type_to_decoder(task_type, embedding_all_dim, feat_class_dim):
    if task_type == "class":
        return nn.Linear(embedding_all_dim, feat_class_dim)
    elif task_type == "time":
        return nn.Sequential(
            nn.Linear(embedding_all_dim, 1),
            nn.ReLU()
        )
    elif task_type == "time_weibull":
        return Weibull_Decoder(embedding_all_dim, 2)


class XTSFormer(torch.nn.Module):
    def __init__(self, device, feat_input_dim, feat_class_dim, decoder, embedding_all_dim, embedding_time_dim,
                 num_layers=3, n_head=2, drop_out=0, gamma=2, total_subseq_len=None):
        super(XTSFormer, self).__init__()

        self.decoder = decoder
        self.feat_input_dim = feat_input_dim
        self.num_layers = num_layers
        self.feat_class_dim = feat_class_dim
        self.embedding_all_dim = embedding_all_dim
        self.embedding_time_dim = embedding_time_dim
        self.device = device
        self.n_head = n_head
        self.gamma = gamma
        self.fc_feat_notime = torch.nn.Linear(feat_input_dim - 2, self.embedding_all_dim - self.embedding_time_dim)
        self.fc_feat_notime_5_19 = torch.nn.Linear(88, self.embedding_all_dim - self.embedding_time_dim)
        self.fc_feat_notime_6_4 = torch.nn.Linear(87, self.embedding_all_dim - self.embedding_time_dim)
        self.embed_activation = torch.nn.LeakyReLU(0.1)
        self.time_encoder = TimeEncode(expand_dim=self.embedding_time_dim)
        self.FCtime_encoder = FCTimeEncode(expand_dim=self.embedding_time_dim,
                                           nontime_feat_dim=self.embedding_all_dim - self.embedding_time_dim)
        self.Mercer_time_encoder = Mercer_TimeEncode(expand_dim=self.embedding_time_dim,
                                           nontime_feat_dim=self.embedding_all_dim - self.embedding_time_dim)
        self.xts_SelfAttention1 = xts_SelfAttention(num_attention_heads=self.n_head, input_size=self.embedding_all_dim,
                                                    hidden_size=self.embedding_all_dim)
        self.xts_SelfAttention2 = xts_SelfAttention(num_attention_heads=self.n_head, input_size=self.embedding_all_dim,
                                                    hidden_size=self.embedding_all_dim)
        self.xts_CrossAttention1 = xts_CrossAttention(num_attention_heads=self.n_head,
                                                      input_size=self.embedding_all_dim,
                                                      hidden_size=self.embedding_all_dim)
        self.MLP = torch.nn.Linear(self.embedding_all_dim, self.feat_class_dim)

    def forward(self, input_data, tree01, tree12, tree23, scale_mask_l0, scale_mask_l1):

        feat_time = input_data[:, :, 2]
        feat_pharmacy_class = input_data[:, :, 1]
        feat_others = xts_tensor_join((input_data[:, :, 0], input_data[:, :, 3]), join_style="new", join_axis=2)
        feat_pharmacy_class = feat_pharmacy_class.view(-1, 1).to("cpu")
        # one_hot = xts_one_hot(feat_pharmacy_class, 86)
        one_hot = xts_one_hot(feat_pharmacy_class, self.feat_class_dim)
        one_hot = one_hot.view(input_data.shape[0], input_data.shape[1], -1)
        feat_notime = xts_tensor_join((one_hot.to(input_data.device), feat_others[:, :, 1:].to(input_data.device)),
                                      join_style="old", join_axis=2).to(input_data.device)
        # embedding_feat_notime = self.fc_feat_notime_5_19(feat_notime)
        embedding_feat_notime = self.fc_feat_notime_6_4(feat_notime.to(input_data.device)).to(input_data.device)


        embedding_time = self.FCtime_encoder(feat_time, embedding_feat_notime)


        x = xts_tensor_join((embedding_feat_notime, embedding_time), join_style="old", join_axis=2)


        # scale masked self attention:
        scale1 = scale_mask_l0[:, :, 0]
        scale2 = scale_mask_l0[:, :, 1]
        scale3 = scale_mask_l0[:, :, 2]
        x_scale1 = self.xts_SelfAttention1(x, scale1)
        x_scale2 = self.xts_SelfAttention1(x, scale2)
        x_scale3 = self.xts_SelfAttention1(x, scale3)

        x_scale1_zero = scale1.unsqueeze(-1).expand(*x_scale1.shape) * x_scale1
        x_scale2_zero = scale2.unsqueeze(-1).expand(*x_scale2.shape) * x_scale2
        x_scale3_zero = scale3.unsqueeze(-1).expand(*x_scale3.shape) * x_scale3
        x = x_scale1_zero + x_scale2_zero + x_scale3_zero


        x_l1 = torch.bmm(tree01, x)

        scale1_2_l2 = scale_mask_l1[:, :, 0]
        scale3_l2 = scale_mask_l1[:, :, 1]
        x_scale1_2 = self.xts_SelfAttention1(x_l1, scale1_2_l2)
        x_scale3 = self.xts_SelfAttention1(x_l1, scale3_l2)

        x_scale1_2_zero = scale1_2_l2.unsqueeze(-1).expand(*x_scale1_2.shape) * x_scale1_2
        x_scale3_zero = scale3_l2.unsqueeze(-1).expand(*x_scale3.shape) * x_scale3
        x_scale1_2 = x_scale1_2_zero + x_scale3_zero

        # x_l1_tilde = self.xts_SelfAttention1(x_l1)
        x_l2 = torch.bmm(tree12, x_scale1_2)
        x_l2_tilde = self.xts_SelfAttention1(x_l2)
        x_l3 = torch.bmm(tree23, x_l2_tilde)
        x_l3_tilde = self.xts_SelfAttention1(x_l3)

        dec_node_list = xts_tensor_join(
            (x[:, -1:, :], x_scale1_2[:, -1:, :], x_l2_tilde[:, -1:, :], x_l3_tilde[:, -1:, :]), join_style="old",
            join_axis=1)

        final_predict = self.xts_CrossAttention1(dec_node_list).squeeze()

        final_predict = self.decoder(final_predict)

        return final_predict


class xts_LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(xts_LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class xts_SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1):
        super(xts_SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.new_input = nn.Linear(input_size, self.all_head_size)
        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        # self.LayerNorm = xts_LayerNorm(hidden_size, eps=1e-12)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask=None):
        # input_tensor: [468669, 5, 4]
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        # mixed_value_layer: [468669, 5, 64]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # value_layer: [468669, 2, 5, 32]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        # attention_scores = attention_scores + attention_mask

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=attention_scores.dtype)  # convert to same dtype
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # add dimensions for heads and query length
            attention_scores = attention_scores + attention_mask * -1e10  # apply mask by adding a large negative value

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + self.new_input(input_tensor))

        return hidden_states


class xts_CrossAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1):
        super(xts_CrossAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.new_input = nn.Linear(input_size, self.all_head_size)
        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        # self.LayerNorm = xts_LayerNorm(hidden_size, eps=1e-12)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        # input_tensor: [468669, 5, 4]
        mixed_query_layer = self.query(input_tensor)

        mixed_query_layer = mixed_query_layer[:, -1, :].unsqueeze(1)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        # mixed_value_layer: [468669, 5, 64]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # value_layer: [468669, 2, 5, 32]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + self.new_input(input_tensor))

        return hidden_states
