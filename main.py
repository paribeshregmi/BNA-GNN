import os
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from dgl.dataloading import DataLoader, ShaDowKHopSampler

from data_loaders.data_loader import load_data
from utils.utils import preprocess_adj, accuracy_bbgcn
from models.models import BBGCN

torch.set_printoptions(edgeitems=10)
np.set_printoptions(precision=2, linewidth=200, suppress=True)

def init_model():
    global g, adj, features, labels, idx_train, idx_test, idx_val
    global train_loader, test_loader, val_loader
    global model, optimizer, arch_optimizer

    args.cuda = torch.cuda.is_available()
    gpu = "cuda:" + str(args.gpu)
    args.device = torch.device(gpu if args.cuda else "cpu")
    print("Training on ", args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs("./saved_models/" + args.dataset, exist_ok=True)
    os.makedirs("./loss_logs/" + args.dataset, exist_ok=True)

    # Load data
    g = load_data(args.dataset)

    train_nids = g.ndata["train_mask"]
    test_nids = g.ndata["test_mask"]
    val_nids = g.ndata["val_mask"]

    if args.fs:
        train_nids = torch.logical_not(torch.logical_or(test_nids, val_nids))

    idx_train = (train_nids == True).nonzero().squeeze()
    idx_test = (test_nids == True).nonzero().squeeze()
    idx_val = (val_nids == True).nonzero().squeeze()

    neighbor_order = [args.neighbor_order] * args.hops

    sampler = ShaDowKHopSampler(neighbor_order)
    train_loader = DataLoader(g, idx_train, sampler, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, num_workers=2)
    test_loader = DataLoader(g, idx_test, sampler, batch_size=args.batch_size, shuffle=True,
                             drop_last=False, num_workers=2)
    val_loader = DataLoader(g, idx_val, sampler, batch_size=args.batch_size, shuffle=True,
                            drop_last=False, num_workers=2)

    # Model and optimizer
    model = BBGCN(args=args,
                nfeat=g.ndata['feat'].shape[1],
                nhid=args.hidden,
                nclass=g.ndata['label'].max().item() + 1,
                dropout=args.dropout,
                device=args.device).to(args.device)

    os.makedirs(f'./saved_models/{args.dataset}', exist_ok=True)
    os.makedirs(f'./loss_logs/{args.dataset}', exist_ok=True)

    if os.path.exists(args.model_file):
        print("Loading saved model from file...")
        model.load_state_dict(torch.load(args.model_file))

    weights = []
    arch_params = []

    for name, param in model.named_parameters():
        if "variational" in name:
            arch_params.append(param)
        else:
            weights.append(param)

    optimizer = optim.Adam(weights, lr=args.lr, weight_decay=args.weight_decay)
    arch_optimizer = optim.Adam(arch_params, lr=args.arch_lr)

    with open("./model_arch.txt", "w") as f:
        f.write(str(model))


def train(epoch):
    model.train()
    scale = 1.0/len(train_loader)
    tot_loss = 0
    tot_kl = 0
    tot_acc = 0
    for _, output_nodes, sub_g in train_loader:
        optimizer.zero_grad()
        arch_optimizer.zero_grad()

        adjacency = preprocess_adj(sub_g.adj()).to(args.device)
        feats = sub_g.ndata['feat'].to(args.device)
        targets = sub_g.ndata['label'].to(args.device)
        sub_train_idx = torch.arange(len(output_nodes), dtype=torch.long)

        output, _ = model(feats, adjacency, args.num_samples)
        filtered_out = output[sub_train_idx].permute(1, 0, 2).reshape(len(sub_train_idx) * args.num_samples, -1)
        filtered_lab = targets[sub_train_idx].repeat(args.num_samples)

        loss_train = F.nll_loss(filtered_out, filtered_lab)
        loss_kl  = scale * model.architecture_sampler.get_kl()

        tot_loss += loss_train.item() * scale
        tot_kl += loss_kl.item() * scale

        loss_train = loss_train + args.kld_weight * loss_kl
        loss_train.backward()
        optimizer.step()
        arch_optimizer.step()

        acc_tr = accuracy_bbgcn(output[sub_train_idx], targets[sub_train_idx])
        tot_acc += acc_tr.item()

    print(f'Epoch: {epoch} --> train_loss: {tot_loss}, kl_loss: {tot_kl}, train_acc: {tot_acc / len(idx_train)}')

def validate():
    model.eval()
    tot_acc = 0
    scale = 1.0 / len(train_loader)
    tot_loss = 0
    for _, output_nodes, sub_g in val_loader:
        adjacency = preprocess_adj(sub_g.adj()).to(args.device)
        feats = sub_g.ndata['feat'].to(args.device)
        targets = sub_g.ndata['label'].to(args.device)
        val_idx = torch.arange(len(output_nodes), dtype=torch.long)

        output, _ = model(feats, adjacency, args.num_samples)
        loss_val = F.nll_loss(output[val_idx,0,:], targets[val_idx])
        tot_loss += loss_val.item() * scale

        acc_val = accuracy_bbgcn(output[val_idx], targets[val_idx])
        tot_acc += acc_val.item()

    tot_acc = tot_acc * 100
    layers, activations, percentile_25, percentile_75 = model.get_arch_info()
    activations = activations.cpu().detach().numpy().mean() * 100

    print(f'val_acc: {tot_acc / len(idx_val)}, percent_activation: {activations}, layers: {layers}')
    print("----------------------------------------------------")

    return tot_acc / len(idx_val)


def test():
    model.load_state_dict(torch.load(args.model_file))
    model.eval()
    tot_acc = 0
    for _, output_nodes, sub_g in test_loader:
        adjacency = preprocess_adj(sub_g.adj()).to(args.device)
        feats = sub_g.ndata['feat'].to(args.device)
        targets = sub_g.ndata['label'].to(args.device)
        sub_test_idx = torch.arange(len(output_nodes), dtype=torch.long)

        output, _ = model(feats, adjacency, args.num_samples)
        acc_test = accuracy_bbgcn(output[sub_test_idx], targets[sub_test_idx])
        tot_acc += acc_test.item()

    tot_acc = tot_acc * 100

    print(f'test_acc: {tot_acc / len(idx_test)}')
    print("----------------------------------------------------")
    return tot_acc / len(idx_test)

def fit(args_l):
    global args
    args = args_l
    init_model()

    bad_counter = 0
    best = 0
    for epoch in range(args.epochs):
        train(epoch)
        with torch.no_grad():
            acc = validate()
        if acc > best:
            best = acc
            torch.save(model.state_dict(), args.model_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    with torch.no_grad():
        return test()