import argparse
import glob
import os
import time

from percon import JointModel
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
    '--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument(
    '--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument(
    '--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument(
    '--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument(
    '--hidden-units', type=str, default="32,32", help="Hidden units in each hidden layer, splitted with comma.")
parser.add_argument(
    '--feature-size', type=str, default="32,32", help="feature size of word and user, splitted with comma.")
parser.add_argument(
    '--heads', type=str, default="8,8,1", help="Heads in each layer, splitted with comma.")
parser.add_argument(
    '--batch', type=int, default=64, help="Batch size.")
parser.add_argument(
    '--patience', type=int, default=10, help="Patience.")
parser.add_argument(
    '--data-dir', type=str, default='input', help="Data file directory.")
parser.add_argument(
    '--pkl-dir', type=str, default='00', help="Model file directory.")
parser.add_argument(
    '--train-ratio', type=float, default=0.8, help="Training ratio (0, 1).")
parser.add_argument(
    '--valid-ratio', type=float, default=0.25, help="Training ratio (0, 1).")
parser.add_argument(
    '--loss-fn', type=str, default='rmse', help="Training loss function (rmse,mae)")
parser.add_argument(
    '--temperature', type=float, default=0.01, help="temperature")
parser.add_argument(
    '--fineturinglr', type=float, default=8e-5, help="temperature")
parser.add_argument(
    '--alpha', type=float, default=0.5, help="temperature")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NPairLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(NPairLoss, self).__init__()

        self.margin = margin

    def forward(self, embeddings, labels):
        n = embeddings.size(0)
        sim_matrix = torch.matmul(embeddings, torch.t(embeddings))
        pos_mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        neg_mask = labels.expand(n, n).ne(labels.expand(n, n).t())

        pos_sim = sim_matrix[pos_mask].view(n, -1)
        neg_sim = sim_matrix[neg_mask].view(n, -1)

        num_pos_pairs = pos_sim.size(1)
        num_neg_pairs = neg_sim.size(1)

        if num_pos_pairs == 0 or num_neg_pairs == 0:
            return torch.tensor(0.0)

        loss = 0.0
        for i in range(n):
            pos_loss = torch.sum(torch.exp(self.margin - pos_sim[i]))
            neg_loss = torch.sum(torch.exp(neg_sim[i] - self.margin))
            loss += torch.log(pos_loss + neg_loss)

        loss /= n

        return loss




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
    # Get the unique labels in the batch
    # print("embeddings1.shape")
    # print(embeddings1.shape)

    # print(embeddings2.shape)
    # print(type(embeddings1))
    # print(type(embeddings1))


    logits = F.cosine_similarity(embeddings1,embeddings2,dim=-1)

    logits /= args.temperature

    loss = -torch.nn.LogSoftmax(0)(logits).diag()

    loss = loss.sum()
    return loss



def train(train_loader, valid_loader, model, optimizer, loss_fn, valid):
    model.train()
    loss_train = np.zeros(5)
    total_train = 0.
    loss1 = 0
    for _, batch in enumerate(train_loader):
        # user_word_adj, sentence_emb, labels = batch
        user,labels = batch

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
        # print(total_train)
        # print(sentence_emb.shape)
        # print(user_word_adj.shape)
        # print("xxxxxxxx")

        output1_new, output2_new, x_temp,attn_pp, attn_wp, x  = model(user_word_adj,sentence_emb)
        # print("output")
        # print(output1.shape)
        # print(output2.shape)
        # loss_train_batch = _get_info_nce_loss(output1,output2).sum(0) / bs
        loss1 = _get_info_nce_loss(output1_new,output2_new)
        # print("info loss")
        # print(loss_train_batch)

        if args.loss_fn == 'rmse':
            loss_train_batch = torch.sqrt(loss_fn(x_temp, labels).sum(0) / bs)
        else:
            loss_train_batch = loss_fn(x_temp, labels).sum(0) / bs
        for i in range(5):
            loss_train[i] += loss_train_batch[i].item()
        total_train += 1
        # todo check sum() or mean() for backward()
        loss = args.alpha * loss1 + (1-args.alpha) * loss_train_batch.mean()
        loss.backward()
        optimizer.step()

    if valid:
        model.eval()
        loss_val = np.zeros(5)
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

            output1_new, output2_new, x_temp,attn_pp, attn_wp, x = model(user_word_adj,sentence_emb)
            loss1 = _get_info_nce_loss(output1_new,output2_new)
            if args.loss_fn == 'rmse':
                loss_val_batch = torch.sqrt(loss_fn(x_temp, labels).sum(0) / bs)
            else:
                loss_val_batch = loss_fn(x_temp, labels).sum(0) / bs
            for i in range(5):
                loss_val[i] += loss_val_batch[i].item()
            total_val += 1
        return (1-args.alpha) *(loss_val.mean() / total_val) +args.alpha * loss1/ total_val, loss
    else:
        return loss




def compute_test(test_loader):
    model.eval()
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
        output1, output2, x_temp,attn_pp, attn_wp, x  = model(user_word_adj,sentence_emb)

        loss1 = _get_info_nce_loss(output1,output2)
        if args.loss_fn == 'rmse':
            loss_test_batch = torch.sqrt(loss_fn(x_temp, labels).sum(0) / bs)
        else:
            loss_test_batch = loss_fn(x_temp, labels).sum(0) / bs
        for i in range(5):
            loss_test[i] += loss_test_batch[i].item()
        total_test += 1
    # print(init_emb.shape)
    # np.save('my250_PerGCN_emb_256.npy', init_emb)
    print('Test duiqi ' + args.loss_fn + ' set loss:{0[0]:.4f} {0[1]:.4f} {0[2]:.4f} {0[3]:.4f} {0[4]:.4f}'
          .format(loss_test / total_test), ' sum:{:.4f}'.format((loss_test / total_test).sum()))
    print('Test  task' + args.loss_fn + ' set loss:{:.4f}'
          .format(loss1 / total_test), ' sum:{:.4f}'.format((loss1 / total_test).sum()))


def draw_k_train_result(k, num_epochs, k_train_loss, k_valid_loss):
    plt.figure(figsize=(25, 5))
    for i in range(k):
        xs = [x for x in range(num_epochs)]
        plt.subplot(1, k, i + 1)
        plt.plot(xs, k_train_loss[i], color='b', label='k_train_loss')
        plt.plot(xs, k_valid_loss[i], color='r', label='k_valid_loss')
        plt.ylim(0, 1.5)
        plt.xlabel('k')
        plt.ylabel('loss')
        plt.title(f'k={i + 1}')
        plt.legend()

    plt.show()
    plt.savefig('./img/{}_{}_{}_{}_{}_{}_{}.png'.
                format(args.data_dir[5:], k, args.epoch, args.lr, args.dropout, args.weight_decay, args.batch))


def k_fold_train(k):
    k_train_loss, k_valid_loss = {}, {}
    final_train_loss, final_valid_loss = [], []

    t_total = time.time()
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, train_x, train_y)
        train_data = TensorDataset(X_train, y_train)
        train_sampler = RandomSampler(train_data)
        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch,drop_last=True)
        valid_data = TensorDataset(X_valid, y_valid)
        valid_sampler = RandomSampler(valid_data)
        valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch,drop_last=True)

        # Model and optimizer
        n_heads = [int(x) for x in args.heads.strip().split(",")]
        model = JointModel(
            pern_feature=pern_feature,
            word_feature=word_feature,
            pern_adj=pern_pern_adj,
            word_pern_adj=word_pern_adj,
            embed_size2 = 768,
            n_units=n_units, n_heads=n_heads,
            word_dim=word_dim, user_dim=user_dim,
            dropout=args.dropout)

        if args.cuda:
            model.cuda()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        params = [{'params': filter(lambda p: p.requires_grad, model.parameters())}]
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        if args.loss_fn == 'rmse':
            loss_fn = torch.nn.MSELoss(reduction='none')
        else:
            loss_fn = torch.nn.L1Loss(reduction='none')
        # Train model
        val_loss_values = []  # val_loss
        train_loss_values = []  # train_loss

        print('Starting {} th fold training...'.format(i))
        with tqdm(total=args.epochs, desc=f'Epoch 0/{args.epochs}') as pbar:
            for epoch in range(args.epochs):
                val_loss, train_loss = train(train_loader, valid_loader, model, optimizer, loss_fn, valid=True)
                val_loss_values.append(val_loss)
                train_loss_values.append(train_loss)

                pbar.set_description_str(f'Epoch {epoch + 1}/{args.epochs}')
                pbar.set_postfix_str('loss_train:{0[0]:.4f} {0[1]:.4f} {0[2]:.4f} {0[3]:.4f} {0[4]:.4f}|| '
                                     'loss_val:{1[0]:.4f} {1[1]:.4f} {1[2]:.4f} {1[3]:.4f} {1[4]:.4f}'.format(
                    train_loss,
                    val_loss))
                pbar.update(1)

        k_train_loss[i] = np.array(train_loss_values).mean(1)
        k_valid_loss[i] = np.array(val_loss_values).mean(1)
        final_train_loss.append(train_loss_values[-1])
        final_valid_loss.append(val_loss_values[-1])

    np.set_printoptions(precision=4)
    print('final train loss:', np.array(final_train_loss).mean(0))
    print('final valid loss:', np.array(final_valid_loss).mean(0),
          'sum:{:.4f}'.format(np.array(final_valid_loss).mean(0).sum()))
    print(f'time:{(time.time() - t_total):<.4f}s')
    return k_train_loss, k_valid_loss


'''
=========k-fold-train==========
'''
# k = 5
# k_train_loss, k_valid_loss = k_fold_train(k)
# draw_k_train_result(k, args.epochs, k_train_loss, k_valid_loss)

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


# user_bert.shape [N ,n*bert.shape -46080]
user_bert = np.loadtxt("input/youtube_bert.txt")
# user_bert = np.loadtxt("input/pan_bert.txt")
# user_bert = np.loadtxt("input/my_bert_embedding.txt")
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
# X. shape  [N,  2393 (-46080+89)]
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


train_x, test_x, train_y, test_y = train_test_split(X, y,
                                                    random_state=args.seed,
                                                    test_size=1 - args.train_ratio)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, random_state=args.seed, test_size=args.valid_ratio)

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
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch, pin_memory=True, num_workers=0,drop_last=True)


train_data = TensorDataset(train_x, train_y)
train_sampler = RandomSampler(train_data)
##修改
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch, pin_memory=True, num_workers=0,drop_last=True)

valid_data = TensorDataset(valid_x, valid_y)
valid_sampler = RandomSampler(valid_data)
#修改
valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch, pin_memory=True, num_workers=0,drop_last=True)

# Model and optimizer
n_heads = [int(x) for x in args.heads.strip().split(",")]
model = JointModel(
    pern_feature=pern_feature,
    word_feature=word_feature,
    pern_adj=pern_pern_adj,
    word_pern_adj=word_pern_adj,
    embed_size2 = 768,
    n_units=n_units, n_heads=n_heads,
    word_dim=word_dim, user_dim=user_dim,
    dropout=args.dropout)

if args.cuda:
    model.cuda()

params = [{'params': filter(lambda p: p.requires_grad, model.parameters())}]
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

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
# best_epoch = 64

print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load(os.path.join(args.pkl_dir, '{}.pkl'.format(best_epoch))))

# 在训练集上测试fine-tuning模型性能
compute_test(test_loader)