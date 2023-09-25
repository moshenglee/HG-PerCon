#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : lml
# @File    : my_train.py
# @Software: PyCharm

import argparse
import glob
import os
import time
from percon import PerCon
from percon import FinetuneModel
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm import tqdm
from torch import nn
from data_loader import InteractionDataSet
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--no-cuda', action='store_true', default=False, help='Disable CUDA training.')
parser.add_argument(
    '--seed', type=int, default=42, help='Random seed.')
parser.add_argument(
    '--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument(
    '--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument(
    '--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument(
    '--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument(
    '--hidden-units', type=str, default="16,16", help="Hidden units in each hidden layer, splitted with comma.")
parser.add_argument(
    '--feature-size', type=str, default="32,32", help="feature size of word and user, splitted with comma.")
parser.add_argument(
    '--heads', type=str, default="16,16,1", help="Heads in each layer, splitted with comma.")
parser.add_argument(
    '--batch', type=int, default=64, help="Batch size.")
parser.add_argument(
    '--patience', type=int, default=10, help="Patience.")
parser.add_argument(
    '--data-dir', type=str, default='input', help="Data file directory.")
parser.add_argument(
    '--pkl-dir', type=str, default='00', help="Model file directory.")
parser.add_argument(
    '--pkl-dir2', type=str, default='00', help="Model file directory.")
parser.add_argument(
    '--train-ratio', type=float, default=0.6, help="Training ratio (0, 1).")
parser.add_argument(
    '--valid-ratio', type=float, default=0.5, help="Training ratio (0, 1).")
parser.add_argument(
    '--loss-fn', type=str, default='rmse', help="Training loss function (rmse,mae)")
parser.add_argument(
    '--temperature', type=float, default=0.02, help="temperature")
parser.add_argument(
    '--fineturinglr', type=float, default=8e-4, help="fineturinglr")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






def train(train_loader, valid_loader, model, optimizer, loss_fn, valid):
    model.train()
    loss_train = np.zeros(5)
    total_train = 0.
    for _, batch in enumerate(train_loader):
        # user_word_adj, sentence_emb, labels = batch
        user,labels = batch
        # print("user_word_adj")

        user_word_adj = user[:,:-46080]
        sentence_emb = user[:,-46080:]


        # print(user_word_adj.shape)
        # print(sentence_emb.shape)
        bs = user_word_adj.size(0)
        if args.cuda:
            user_word_adj = user_word_adj.cuda()
            sentence_emb = sentence_emb.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        output1, output2, output1_new, output2_new,attn_pp, attn_wp, x = model(user_word_adj,sentence_emb)

        # loss_train_batch = _get_info_nce_loss(output1,output2).sum(0) / bs
        loss_train_batch = _get_info_nce_loss(output1_new,output2_new)/bs

        # if args.loss_fn == 'rmse':
        #     loss_train_batch = torch.sqrt(loss_fn(output, labels).sum(0) / bs)
        # else:
        #     loss_train_batch = loss_fn(output, labels).sum(0) / bs
        # for i in range(5):
        #     loss_train[i] += loss_train_batch[i].item()
        total_train += 1
        # todo check sum() or mean() for backward()
        loss_train_batch.mean().backward()
        optimizer.step()
    if valid:
        model.eval()
        loss_val = 0
        total_val = 0.
        for _, batch in enumerate(valid_loader):
            user,labels = batch


            user_word_adj = user[:,:-46080]
            sentence_emb = user[:,-46080:]


            # user_word_adj, sentence_emb, labels = batch
            bs = user_word_adj.size(0)

            if args.cuda:
                user_word_adj = user_word_adj.cuda()
                sentence_emb = sentence_emb.cuda()
                labels = labels.cuda()
            # print(user_word_adj.shape)
            # print(sentence_emb.shape)
            output1, output2, output1_new, output2_new,attn_pp, attn_wp, x = model(user_word_adj,sentence_emb)
            loss_val_batch = _get_info_nce_loss(output1_new,output2_new).sum(0) / bs
            loss_val += loss_val_batch.item()
            # if args.loss_fn == 'rmse':
            #     loss_val_batch = torch.sqrt(loss_fn(output, labels).sum(0) / bs)
            # else:
            #     loss_val_batch = loss_fn(output, labels).sum(0) / bs
            # for i in range(5):
            #     loss_val[i] += loss_val_batch[i].item()
            total_val += 1
            loss_val_batch.mean().backward()
            optimizer.step()
        return loss_val / total_val, loss_train / total_train
    else:
        return loss_train / total_train


def train2(train_loader, valid_loader, model, optimizer, loss_fn, valid):
    model.train()
    list_train = []
    loss_train = np.zeros(5)
    total_train = 0.
    for _, batch in enumerate(train_loader):
        user,labels = batch


        user_word_adj = user[:,:-46080]
        sentence_emb = user[:,-46080:]


        bs = user_word_adj.size(0)
        if args.cuda:
            user_word_adj = user_word_adj.cuda()
            sentence_emb = sentence_emb.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()

        output1, output2,x_temp, output1_new, output2_new,attn_pp, attn_wp, x  = model(user_word_adj,sentence_emb)
        # user_word_adj1 = user_word_adj.cpu().detach().numpy().tolist()
        # sentence_emb1 = sentence_emb.cpu().detach().numpy().tolist()
        # x_temp1 = x_temp.cpu().detach().numpy().tolist()
        # list_train_temp=[user_word_adj1,sentence_emb1,x_temp1]
        # list_train.append(list_train_temp)

        if args.loss_fn == 'rmse':
            loss_train_batch = torch.sqrt(loss_fn(x_temp, labels).sum(0) / bs)
        else:
            loss_train_batch = loss_fn(x_temp, labels).sum(0) / bs
        for i in range(5):
            loss_train[i] += loss_train_batch[i].item()
        total_train += 1
        # todo check sum() or mean() for backward()
        # print(loss_train_batch)
        loss_train_batch.requires_grad_(True)

        loss_train_batch.mean().backward()
        optimizer.step()
    # print(type(user_word_adj))
    # print(type(x_temp1))
    # # print(list_train)
    # b=np.array(list_train).reshape(-1,3)
    # print(b[0])
    # # print(b.shape)
    # np.savetxt(r'trainembedding.txt', b)
    # np.savetxt("trainembedding.txt",b, delimiter=',', fmt='%.8f')
    if valid:
        model.eval()
        loss_val = np.zeros(5)
        total_val = 0.
        for _, batch in enumerate(valid_loader):
            user,labels = batch


            user_word_adj = user[:,:-46080]
            sentence_emb = user[:,-46080:]

            bs = user_word_adj.size(0)

            if args.cuda:
                user_word_adj = user_word_adj.cuda()
                sentence_emb = sentence_emb.cuda()
                labels = labels.cuda()

            output1, output2,x_temp, output1_new, output2_new,attn_pp, attn_wp, x = model(user_word_adj,sentence_emb)
            if args.loss_fn == 'rmse':
                loss_val_batch = torch.sqrt(loss_fn(x_temp, labels).sum(0) / bs)
            else:
                loss_val_batch = loss_fn(x_temp, labels).sum(0) / bs
            for i in range(5):
                loss_val[i] += loss_val_batch[i].item()
            total_val += 1
        return loss_val / total_val, loss_train / total_train
    else:
        return loss_train / total_train




def _get_info_nce_loss(embeddings1, embeddings2):
    """
    Calculates the multi-class N-pair contrastive loss.

    Args:
        labels: A tensor of shape (batch_size,) containing the true labels for each example.
        embeddings: A tensor of shape (batch_size, embedding_size) containing the embeddings for each example.
        alpha: A float representing the weight of the negative samples.

    Returns:
        The multi-class N-pair contrastive loss.
    """
    batch_size = embeddings1.shape[0]

    logits = F.cosine_similarity(embeddings1,embeddings2,dim=-1)

    logits /= args.temperature

    loss = -torch.nn.LogSoftmax(0)(logits).diag()

    loss = loss.sum()
    return loss

def compute_test(test_loader):
    fine_tuning_model.eval()
    loss_test = np.zeros(5)
    total_test = 0.
    for idx, batch in enumerate(test_loader):
        # user_word_adj, labels = batch
        user,labels = batch

        user_word_adj = user[:,:-46080]
        sentence_emb = user[:,-46080:]

        bs = user_word_adj.size(0)

        if args.cuda:
            user_word_adj = user_word_adj.cuda()
            sentence_emb = sentence_emb.cuda()
            labels = labels.cuda()

        # output, attn_pp, attn_wp, x = model(user_word_adj)
        output1, output2, x_temp, output1_new, output2_new,attn_pp, attn_wp, x = fine_tuning_model(user_word_adj,sentence_emb)
        if idx == 0:
            # save attention weights
            np.save('./visual/output1_new.npy', output1_new.cpu().detach().numpy())
            np.save('./visual/output2_new.npy', output2_new.cpu().detach().numpy())
            np.save('./visual/user_word_adj.npy', user_word_adj.cpu().detach().numpy())
            np.save('./visual/sentence_emb.npy', sentence_emb.cpu().detach().numpy())
            np.save('./visual/output1.npy', output1.cpu().detach().numpy())
            np.save('./visual/output2.npy', output2.cpu().detach().numpy())


            init_emb = x.cpu().detach().numpy()
        else:
            init_emb = np.concatenate((init_emb, x.cpu().detach().numpy()), axis=0)
        if args.loss_fn == 'rmse':
            loss_test_batch = torch.sqrt(loss_fn(x_temp, labels).sum(0) / bs)
        else:
            loss_test_batch = loss_fn(x_temp, labels).sum(0) / bs
        for i in range(5):
            loss_test[i] += loss_test_batch[i].item()


        total_test += 1

    print('Test ' + args.loss_fn + ' set loss:{0[0]:.4f} {0[1]:.4f} {0[2]:.4f} {0[3]:.4f} {0[4]:.4f}'
          .format(loss_test / total_test), ' sum:{:.4f}'.format((loss_test / total_test).sum()))




'''
=========True training=========
train:valid:test  0.25:0.25:0.5
get 10 times avg scores
'''

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.pkl_dir):
    os.makedirs(args.pkl_dir)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataset = InteractionDataSet(args.data_dir)

N = len(dataset)
feature_dim = dataset.get_feature_dimension()
word_dim, user_dim = [int(x) for x in args.feature_size.strip().split(",")]
n_units = [int(x) for x in args.hidden_units.strip().split(",")]


user_bert = np.loadtxt("input/youtube_bert.txt")
print("user_bert.shape")
print(user_bert.shape)

word_pern_adj = dataset.get_word_pern_adj()
pern_pern_adj = dataset.get_pern_adj()
pern_feature = dataset.get_pern_features()
word_feature = dataset.get_word_features()
# X = dataset.get_user_word_adj()
X1 = dataset.get_user_word_adj()
X = torch.cat((torch.tensor(X1),torch.tensor(user_bert)),1)
print("X.shape")
print(X.shape)
# X. shape  [N,  2393 (2304+89)]
print("======0000=====")
y = dataset.get_labels()
word_pern_adj = torch.tensor(word_pern_adj)
pern_pern_adj = torch.tensor(pern_pern_adj)
pern_feature = torch.tensor(pern_feature)
word_feature = torch.tensor(word_feature)
if args.cuda:
    word_pern_adj = torch.tensor(word_pern_adj)
    pern_pern_adj = torch.tensor(pern_pern_adj)
    pern_feature = torch.tensor(pern_feature)
    word_feature = torch.tensor(word_feature)
    print(word_pern_adj)


    word_pern_adj = word_pern_adj.cuda()
    pern_pern_adj = pern_pern_adj.cuda()
    pern_feature = pern_feature.cuda()
    word_feature = word_feature.cuda()



# valid_ratio = 0.20
test_size = 1 - args.train_ratio
print(test_size)
train_x, test_x1, train_y, test_y1 = train_test_split(X, y,
                                                      random_state=args.seed,
                                                      test_size=1 - args.train_ratio)
valid_x, test_x, valid_y,test_y = train_test_split(test_x1, test_y1, random_state=args.seed, test_size=args.valid_ratio)

##修改
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)
test_x = torch.tensor(test_x)
test_y = torch.tensor(test_y)
valid_x = torch.tensor(valid_x)
valid_y = torch.tensor(valid_y)
print('train_x size:', train_x.shape, 'train_y size:', train_y.shape)
print('test_x size:', test_x.shape, 'test_y size:', test_y.shape)
print('valid_x size:', valid_x.shape, 'valid_y size:', valid_y.shape)

test_data = TensorDataset(test_x, test_y)
test_sampler = RandomSampler(test_x)
##修改
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch, pin_memory=True, num_workers=0)


train_data = TensorDataset(train_x, train_y)
train_sampler = RandomSampler(train_data)
##修改
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch, pin_memory=True, num_workers=0)

valid_data = TensorDataset(valid_x, valid_y)
valid_sampler = RandomSampler(valid_data)
#修改
valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch, pin_memory=True, num_workers=0)

# Model and optimizer
n_heads = [int(x) for x in args.heads.strip().split(",")]
model = PerCon(
    pern_feature=pern_feature,
    word_feature=word_feature,
    pern_adj=pern_pern_adj,
    word_pern_adj=word_pern_adj,
    embed_size = 768,
    n_units=n_units, n_heads=n_heads,
    word_dim=word_dim, user_dim=user_dim,
    dropout=args.dropout)

if args.cuda:
    model.cuda()

params = [{'params': filter(lambda p: p.requires_grad, model.parameters())}]
optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
if args.loss_fn == 'rmse':
    loss_fn = torch.nn.MSELoss(reduction='none')
else:
    loss_fn = torch.nn.L1Loss(reduction='none')

bad_counter = 0
best = args.epochs + 1
best_val = []
best_epoch = 0
loss_train = []
loss_val = []
t_total = time.time()
# print('torch cuda device count',torch.cuda.device_count())
# print('torch cuda device name', torch.cuda.get_device_name())
files = glob.glob(os.path.join(args.pkl_dir, '*.pkl'))
for file in files:
    os.remove(file)
with tqdm(total=args.epochs, desc=f'Epoch 0/{args.epochs}') as pbar:
    for epoch in range(args.epochs):
        val_loss, train_loss = train(train_loader, valid_loader, model, optimizer, loss_fn, True)
        loss_train.append(train_loss)
        loss_val.append(val_loss)
        torch.save(model.state_dict(), os.path.join(args.pkl_dir, '{}.pkl'.format(epoch)))
        # todo check sum() or mean()
        if loss_val[-1] < best:
            best = loss_val[-1]
            best_val = loss_val[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
        pbar.set_description_str(f'Epoch {epoch + 1}/{args.epochs}')
        pbar.set_postfix_str('loss_train:{} || '
                             'loss_val:{} '.format(train_loss, val_loss))
        pbar.update(1)
        files = glob.glob(os.path.join(args.pkl_dir, '*.pkl'))
        for file in files:
            epoch_nb = int(file.split('/')[-1].split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

files = glob.glob(os.path.join(args.pkl_dir, '*.pkl'))
for file in files:
    epoch_nb = int(file.split('/')[-1].split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print('best_val_loss:', best_val)
'''
Restore best model for testing
'''
# 取 youtube_new模型的最佳epoch

print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load(os.path.join(args.pkl_dir, '{}.pkl'.format(best_epoch))))

# 设置fine-tuning模型特征提取器部分的参数不需要更新
for param in model.parameters():
    param.requires_grad = False

fine_tuning_model = FinetuneModel(model)
fine_tuning_model=fine_tuning_model.to(device)


for param in fine_tuning_model.pretrain_model.parameters():
    param.requires_grad = False

for param in fine_tuning_model.fc3.parameters():
    param.requires_grad = True


# 使用预训练模型的参数初始化fine-tuning模型
# fine_tuning_model.load_state_dict(torch.load(os.path.join(args.pkl_dir, '{}.pkl'.format(best_epoch))))


# 定义fine-tuning优化器和损失函数
fine_tuning_optimizer = optim.Adam(fine_tuning_model.fc3.parameters(), lr=args.fineturinglr)
# fine_tuning_criterion = nn.CrossEntropyLoss()
# if args.loss_fn == 'rmse':
#     loss_train_batch = torch.sqrt(loss_fn(output, labels).sum(0) / bs)
# else:
#     loss_train_batch = loss_fn(output, labels).sum(0) / bs
# fine-tuning模型训练
# compiletrain()


if args.loss_fn == 'rmse':
    loss_fn = torch.nn.MSELoss(reduction='none')
else:
    loss_fn = torch.nn.L1Loss(reduction='none')

bad_counter2 = 0
best2 = args.epochs + 1
best_val2 = []
best_epoch2 = 0
loss_train2 = []
loss_val2 = []
t_total2 = time.time()
# print('torch cuda device count',torch.cuda.device_count())
# print('torch cuda device name', torch.cuda.get_device_name())
files2 = glob.glob(os.path.join(args.pkl_dir2, '*.pkl'))
for file in files2:
    os.remove(file)
with tqdm(total=args.epochs, desc=f'Epoch 0/{args.epochs}') as pbar:
    for epoch in range(args.epochs):

        val_loss2, train_loss2 = train2(train_loader, valid_loader, fine_tuning_model, fine_tuning_optimizer, loss_fn, True)
        loss_train2.append(train_loss2)
        loss_val2.append(val_loss2)
        torch.save(fine_tuning_model.state_dict(), os.path.join(args.pkl_dir2, '{}.pkl'.format(epoch)))
        # todo check sum() or mean()
        if loss_val2[-1].mean() < best2:
            best2 = loss_val2[-1].mean()
            best_val2 = loss_val2[-1]
            best_epoch2 = epoch
            bad_counter2 = 0
        else:
            bad_counter2 += 1

        if bad_counter2 == args.patience:
            break
        pbar.set_description_str(f'Epoch {epoch + 1}/{args.epochs}')
        pbar.set_postfix_str('loss_train:{0[0]:.4f} {0[1]:.4f} {0[2]:.4f} {0[3]:.4f} {0[4]:.4f}|| '
                             'loss_val:{1[0]:.4f} {1[1]:.4f} {1[2]:.4f} {1[3]:.4f} {1[4]:.4f}'.format(train_loss2,
                                                                                                      val_loss2))
        pbar.update(1)
        files = glob.glob(os.path.join(args.pkl_dir2, '*.pkl'))
        for file in files:
            epoch_nb = int(file.split('/')[-1].split('.')[0])
            if epoch_nb < best_epoch2:
                os.remove(file)

files2 = glob.glob(os.path.join(args.pkl_dir2, '*.pkl'))
for file in files2:
    epoch_nb = int(file.split('/')[-1].split('.')[0])
    if epoch_nb > best_epoch2:
        os.remove(file)
print("fine-tuning train Optimization Finished!")
print("fine-tuning Total time elapsed: {:.4f}s".format(time.time() - t_total2))
print('best_val_loss:', best_val2)

print('Loading {}th epoch'.format(best_epoch2))
fine_tuning_model.load_state_dict(torch.load(os.path.join(args.pkl_dir2, '{}.pkl'.format(best_epoch2))))

# 在训练集上测试fine-tuning模型性能
compute_test(test_loader)

