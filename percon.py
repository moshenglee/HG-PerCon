import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from gat_layer import MultiHeadGraphAttention


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PerCon(nn.Module):
    def __init__(self, pern_feature, word_feature,
                 pern_adj, word_pern_adj,embed_size2,
                 n_units=None, n_heads=None,
                 word_dim=32, user_dim=32,
                 dropout=0.1,
                 attn_dropout=0 ):

        super(PerCon, self).__init__()
        if n_units is None:
            n_units = [32, 32]
        if n_heads is None:
            n_heads = [8, 8, 1]
        self.dropout = dropout
        self.embed_size2 = embed_size2
        f_word, f_user = word_dim, user_dim
        n_units = n_units + [f_user]
        self.pern_feature = pern_feature

        self.word_feature = nn.Parameter(word_feature)
        self.pern_adj = pern_adj
        self.word_pern_adj = word_pern_adj
        self.gat1_pp = MultiHeadGraphAttention(n_heads[0], f_in=n_units[0] * n_heads[0],
                                               f_out=n_units[1], attn_dropout=attn_dropout)

        self.gat2_wp = MultiHeadGraphAttention(n_heads[1], f_in=n_units[1] * n_heads[1],
                                               f_out=n_units[2], attn_dropout=attn_dropout)
        # self.fc = nn.Linear(n_units[-1], 5)

        self.pt = TransformerEncoder(input_dim=768, hidden_dim=embed_size2, num_layers=2, num_heads=1, dropout_rate=0.1)
        # self.fc1 = nn.Linear(640, 128)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(embed_size2, 128)
        # train
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(128, 64)

    def forward(self,user_word_adj, sentence_embed):


        n_p = self.pern_feature.shape[0]

        h1, attn_pp = self.gat1_pp(self.pern_feature, self.pern_adj)
        h1 = F.elu(h1.transpose(0, 1).contiguous().view(n_p, -1))
        h1 = F.dropout(h1, self.dropout, training=self.training)
        h1_wp = torch.cat((self.word_feature, h1), 0)

        # check only use wp gat
        # self.pern_feature = F.dropout(F.elu(self.pern_feature),self.dropout,training=self.training)
        # h1_wp = torch.cat((self.word_feature, self.pern_feature), 0)

        h2, attn_wp = self.gat2_wp(h1_wp, self.word_pern_adj)  # ([8, 154, 16])
        h2 = h2.mean(dim=0)  # ([154, 16])
        h2_word = h2[:self.word_feature.size(0), :]  # ([101, 16])

        # check only use pp gat
        # h2_word = h1_wp[:self.word_feature.size(0), :]
        # print(user_word_adj[:,:89].shape)
        # print(h2_word.shape)
        x = torch.mm(user_word_adj[:,:89], h2_word)
        x_c = torch.cat((x, user_word_adj[:, 89:]), dim=1)
        # print(x_c.shape)
        x = torch.div(x_c, x_c.sum(1).unsqueeze(1))
        # print(x.shape)
        # x_c = torch.cat((x, user_word_adj[:, 89:]), dim=1)
        output1 = self.fc1(x)
        # output1 = self.fc3(output1)
        # output1 shape[]

        # x2 shape [N, m (post count), bert.shape]

        sentence_embed = self.pt(sentence_embed)
        sentence_embed = torch.mean(sentence_embed, dim=1)


        output2 = self.fc2(sentence_embed)
        output1_new = self.fc4(output1)
        output2_new = self.fc4(output2)
        # output2 = self.fc4(output2)
        # output2 = self.dropout2( self.norm2(self.TTransformer(sentence_embed, sentence_embed, sentence_embed, t) + sentence_embed))
        return output1, output2, output1_new, output2_new,attn_pp, attn_wp, x



# fine-turning
class FinetuneModel(nn.Module):
    def __init__(self, pretrain_model):
        super(FinetuneModel, self).__init__()
        self.pretrain_model = pretrain_model
        self.fc3 = nn.Sequential(
            nn.Linear(128*2, 128),

            nn.ReLU(inplace=True),nn.Linear(128, 5))


        # self.fc3 = nn.Sequential(
        #            nn.Linear(128*2, 128),
        #            nn.ReLU(inplace=True),
        #            nn.Linear(128, 64),
        #            nn.ReLU(inplace=True),nn.Linear(64, 5))
        # self.fc3 = nn.Sequential(
        #            nn.Linear(64*2, 64),
        #            nn.ReLU(inplace=True),
        #            nn.Linear(64, 32),
        #            nn.ReLU(inplace=True),nn.Linear(32, 5))


    def forward(self, x1, x2):
        output1, output2, output1_new, output2_new,attn_pp, attn_wp, x = self.pretrain_model(x1,x2)
        # x2 = self.pretrain_model(x2)
        # print(output1)
        # print(output2)
        # print("sdasdasdas")
        # print(output1.shape)
        # print(output2.shape)
        # print(output1.is_cuda)
        # print(output2.is_cuda)
        output1 = output1.to(device)
        output2 = output2.to(device)
        output1_new = output1_new.to(device)
        output2_new = output2_new.to(device)

        x_temp = torch.cat([output1, output2], dim=1)
        x_temp = self.fc3(x_temp)
        return output1, output2,x_temp, output1_new, output2_new,attn_pp, attn_wp, x



class JointModel(nn.Module):
    def __init__(self, pern_feature, word_feature,
                 pern_adj, word_pern_adj,embed_size2,
                 n_units=None, n_heads=None,
                 word_dim=32, user_dim=32,
                 dropout=0.1,
                 attn_dropout=0 ):
        super(JointModel, self).__init__()
        if n_units is None:
            n_units = [32, 32]
        if n_heads is None:
            n_heads = [8, 8, 1]
        self.dropout = dropout
        self.embed_size2 = embed_size2
        f_word, f_user = word_dim, user_dim
        n_units = n_units + [f_user]
        self.pern_feature = pern_feature

        self.word_feature = nn.Parameter(word_feature)
        self.pern_adj = pern_adj
        self.word_pern_adj = word_pern_adj
        self.gat1_pp = MultiHeadGraphAttention(n_heads[0], f_in=n_units[0] * n_heads[0],
                                               f_out=n_units[1], attn_dropout=attn_dropout)

        self.gat2_wp = MultiHeadGraphAttention(n_heads[1], f_in=n_units[1] * n_heads[1],
                                               f_out=n_units[2], attn_dropout=attn_dropout)
        # self.fc = nn.Linear(n_units[-1], 5)

        # self.fc1 = nn.Linear(640, 128)
        self.share1 = nn.Linear(256, 128)
        self.share2 = nn.Linear(embed_size2, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(128, 64)
        # self.fc = nn.Linear(
        #           128*2,5)
        self.fc = nn.Sequential(
            nn.Linear(128*2, 128),
            nn.ReLU(inplace=True),nn.Linear(128, 5))



    def forward(self, user_word_adj, sentence_embed):
        n_p = self.pern_feature.shape[0]
        h1, attn_pp = self.gat1_pp(self.pern_feature, self.pern_adj)
        h1 = F.elu(h1.transpose(0, 1).contiguous().view(n_p, -1))
        h1 = F.dropout(h1, self.dropout, training=self.training)
        h1_wp = torch.cat((self.word_feature, h1), 0)

        # check only use wp gat
        # self.pern_feature = F.dropout(F.elu(self.pern_feature),self.dropout,training=self.training)
        # h1_wp = torch.cat((self.word_feature, self.pern_feature), 0)

        h2, attn_wp = self.gat2_wp(h1_wp, self.word_pern_adj)  # ([8, 154, 16])
        h2 = h2.mean(dim=0)  # ([154, 16])
        h2_word = h2[:self.word_feature.size(0), :]  # ([101, 16])

        # check only use pp gat
        # h2_word = h1_wp[:self.word_feature.size(0), :]
        # print(user_word_adj[:,:89].shape)
        # print(h2_word.shape)
        x = torch.mm(user_word_adj[:,:89], h2_word)
        x_c = torch.cat((x, user_word_adj[:, 89:]), dim=1)

        # print("error")
        # print(x.shape)
        # print(user_word_adj[:, 89:].shape)
        # print(x_c.shape)
        x = x.to(device)
        x_c = x_c.to(device)
        x = torch.div(x_c, x_c.sum(1).unsqueeze(1))
        # print("end")

        # x_c = torch.cat((x, user_word_adj[:, 89:]), dim=1)
        output1 = self.share1(x)
        # output1 = self.fc3(output1)
        # output1 shape[]

        # x2 shape [N, 4* (post count), bert.shape]
        # x2 = torch.flatten(sentence_embed, 1)
        # x2 shape [N, m (post count)* bert.shape]
        output2 = self.share2(sentence_embed)
        # task1_output1, task1_output2 = self.task1_layer(output1,output2)
        output1_new = self.fc1(output1)
        output2_new = self.fc1(output2)

        x_temp = torch.cat([output1, output2], dim=1)
        x_temp = self.fc(x_temp)


        return output1_new, output2_new, x_temp,attn_pp, attn_wp, x



class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate):
        super(TransformerEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.layer_norm_in = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.linear_out = nn.Linear(hidden_dim, input_dim)
        self.layer_norm_out = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.layer_norm_in(x)

        x = self.transformer_encoder(x)

        x = self.linear_out(x)
        x = self.layer_norm_out(x)
        return x

