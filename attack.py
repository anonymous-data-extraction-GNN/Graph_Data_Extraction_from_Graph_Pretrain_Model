import argparse

from loader import MoleculeDatasetForAttack

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

#from model import GNN
from model import GNN
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from util import ExtractSubstructureContextPair

from dataloader import DataLoaderSubstructContext

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from forgetting_regs import feature_regs, L2_SP

from tensorboardX import SummaryWriter

def pool_func(x, batch, mode = "sum"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

criterion = nn.BCEWithLogitsLoss()

def train(args, model_substruct, model_context, loader, optimizer_substruct, optimizer_context, device, forgetting_reg_substruct, forgetting_reg_context):
    model_substruct.train()
    model_context.train()

    balanced_loss_accum = 0
    acc_accum = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        # creating substructure representation
        substruct_rep = model_substruct(batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct)[batch.center_substruct_idx]

        ### creating context representations
        overlapped_node_rep = model_context(batch.x_context, batch.edge_index_context, batch.edge_attr_context)[batch.overlap_context_substruct_idx]

        # positive context representation
        context_rep = pool_func(overlapped_node_rep, batch.batch_overlapped_context, mode=args.context_pooling)

        pred = torch.sum(substruct_rep * context_rep, dim = 1)
        pred_pos = pred[batch.y==1]
        pred_neg = pred[batch.y==0]

        loss_pos = criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())

        loss_forget_substruct = forgetting_reg_substruct(batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct)
        loss_forget_context = forgetting_reg_context(batch.x_context, batch.edge_index_context, batch.edge_attr_context)
        loss_forget = loss_forget_substruct + loss_forget_context

        optimizer_substruct.zero_grad()
        optimizer_context.zero_grad()

        loss = loss_pos + args.neg_samples * loss_neg + args.forget_strength * loss_forget
        loss.backward()
        #To write: optimizer
        optimizer_substruct.step()
        optimizer_context.step()

        balanced_loss_accum += float(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item())
        acc_accum += (float(torch.sum(pred_pos > 0).detach().cpu().item()) + float(torch.sum(pred_neg < 0).detach().cpu().item())) / (len(pred_pos) + len(pred_neg))

    return balanced_loss_accum/(step+1), acc_accum/(step+1)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--csize', type=int, default=3,
                        help='context size (default: 3).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--neg_samples', type=int, default=1,
                        help='number of negative contexts per positive context (default: 1)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--context_pooling', type=str, default="mean",
                        help='how the contexts are pooled (sum, mean, or max)')
    # parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')

    # my parameters
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--scaffold_smiles', type=str, default='O=C(NC1=CC=CC=C1)C1=CC=CC=C1')
    parser.add_argument("--forgetting_type", type=str, choices=['L2_SP', "feature"], default="L2_SP")
    parser.add_argument("--forget_strength", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    l1 = args.num_layer - 1
    l2 = l1 + args.csize

    print("num layer: %d l1: %d l2: %d" %(args.num_layer, l1, l2))

    #set up dataset and transform function.
    dataset_pos = MoleculeDatasetForAttack(f"dataset/attack/{args.scaffold_smiles}/pos", dataset=None, transform = ExtractSubstructureContextPair(args.num_layer, l1, l2))
    dataset_neg = MoleculeDatasetForAttack(f"dataset/attack/{args.scaffold_smiles}/neg", dataset=None, transform = ExtractSubstructureContextPair(args.num_layer, l1, l2))

    dataset = MoleculeDatasetForAttack(f"dataset/attack/{args.scaffold_smiles}", dataset=None, transform = ExtractSubstructureContextPair(args.num_layer, l1, l2))
    loader = DataLoaderSubstructContext(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    # 平衡正负样本比
    args.neg_samples = len(dataset_pos) / len(dataset_neg)

    # set up models, one for pre-training and one for context embeddings
    model_substruct_origin, model_substruct = [GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                          gnn_type=args.gnn_type).to(device) for _ in range(2)]
                        
    model_context_origin, model_context = [GNN(int(l2-l1), args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                        gnn_type=args.gnn_type).to(device) for _ in range(2)]

    state = torch.load(args.pretrained_model_pth,
                       map_location='cpu')

    model_substruct.load_state_dict(state['model_substruct'])
    model_substruct_origin.load_state_dict(state['model_substruct'])
    model_context.load_state_dict(state['model_context'])
    model_context_origin.load_state_dict(state['model_context'])

    for param in model_substruct_origin.parameters():
        param.requires_grad = False 
    for param in model_context_origin.parameters():
        param.requires_grad = False

    #set up optimizer for the two GNNs
    optimizer_substruct = optim.Adam(model_substruct.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_context = optim.Adam(model_context.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.forgetting_type == "L2_SP":
        forgetting_reg_substruct = L2_SP(model_substruct_origin, model_substruct)
        forgetting_reg_context = L2_SP(model_context_origin, model_context)
    elif args.forgetting_type == "feature":
        forgetting_reg_substruct = feature_regs(model_substruct_origin, model_substruct, [f"gnns.{i}"  for i in range(args.num_layer)])
        forgetting_reg_context = feature_regs(model_context_origin, model_context, [f"gnns.{i}"  for i in range(int(l2-l1))])


    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_loss, train_acc = train(args, model_substruct, model_context, loader, optimizer_substruct, optimizer_context, device, forgetting_reg_substruct, forgetting_reg_context)
        print(train_loss, train_acc)

    if not args.output_model_file == "":
        torch.save(model_substruct.state_dict(), args.output_model_file + ".pth")

if __name__ == "__main__":
    #cycle_index(10,2)
    main()
