# from config import args
# from util import get_num_task

# from dataloaders import MoleculeDataset
from os.path import join
import os
import json as _json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from config import args
from sklearn.metrics import (roc_auc_score)

from dataloaders.splitters import random_scaffold_split, random_split, scaffold_split
from torch.utils.data import DataLoader
# from util import get_num_task

from dataloaders import MoleculeDatasetRich

# from model import GinT5TransformerForConditionalGeneration
from model import GraphormerModel, GraphormerConfig,GinConfig,KVPLMConfig,Graphormer_version_dict, get_graph_model
from dataloaders import GraphData_collator

from transformers import (
    HfArgumentParser,
)
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()

# about seed and basic infod
# parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--no_cuda',action='store_true')

# about dataset and dataloader
parser.add_argument('--input_data_dir', type=str, default='')
parser.add_argument('--dataset', type=str, default='bace')
parser.add_argument('--num_workers', type=int, default=4)

# about training strategies
parser.add_argument('--split', type=str, default='scaffold')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--grad_accum_step',type=int,default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_scale', type=float, default=1)
parser.add_argument('--decay', type=float, default=0)

# about molecule GNN
# parser.add_argument('--gnn_type', type=str, default='gin')
# parser.add_argument('--num_layer', type=int, default=5)
# parser.add_argument('--emb_dim', type=int, default=300)
# parser.add_argument('--dropout_ratio', type=float, default=0.5)

parser.add_argument('--JK', type=str, default='last')
parser.add_argument('--gnn_lr_scale', type=float, default=1)
parser.add_argument('--model_3d', type=str, default='schnet', choices=['schnet'])

# for AttributeMask
parser.add_argument('--mask_rate', type=float, default=0.15)
parser.add_argument('--mask_edge', type=int, default=0)

# for ContextPred
parser.add_argument('--csize', type=int, default=3)
parser.add_argument('--contextpred_neg_samples', type=int, default=1)

# for SchNet
parser.add_argument('--num_filters', type=int, default=128)
parser.add_argument('--num_interactions', type=int, default=6)
parser.add_argument('--num_gaussians', type=int, default=51)
parser.add_argument('--cutoff', type=float, default=10)
parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'add'])
parser.add_argument('--schnet_lr_scale', type=float, default=1)

# for 2D-3D Contrastive CL
parser.add_argument('--CL_neg_samples', type=int, default=1)
parser.add_argument('--CL_similarity_metric', type=str, default='InfoNCE_dot_prod',
                    choices=['InfoNCE_dot_prod', 'EBM_dot_prod'])
parser.add_argument('--T', type=float, default=0.1)
parser.add_argument('--normalize', dest='normalize', action='store_true')
parser.add_argument('--no_normalize', dest='normalize', action='store_false')
parser.add_argument('--SSL_masking_ratio', type=float, default=0)
# This is for generative SSL.
parser.add_argument('--AE_model', type=str, default='AE', choices=['AE', 'VAE'])
parser.set_defaults(AE_model='AE')

# for 2D-3D AutoEncoder
parser.add_argument('--AE_loss', type=str, default='l2', choices=['l1', 'l2', 'cosine'])
parser.add_argument('--detach_target', dest='detach_target', action='store_true')
parser.add_argument('--no_detach_target', dest='detach_target', action='store_false')
parser.set_defaults(detach_target=True)

# for 2D-3D Variational AutoEncoder
parser.add_argument('--beta', type=float, default=1)

# for 2D-3D Contrastive CL and AE/VAE
parser.add_argument('--alpha_1', type=float, default=1)
parser.add_argument('--alpha_2', type=float, default=1)

# for 2D SSL and 3D-2D SSL
parser.add_argument('--SSL_2D_mode', type=str, default='AM')
parser.add_argument('--alpha_3', type=float, default=0.1)
parser.add_argument('--gamma_joao', type=float, default=0.1)
parser.add_argument('--gamma_joaov2', type=float, default=0.1)

# about if we would print out eval metric for training data
parser.add_argument('--eval_train', dest='eval_train', action='store_true')
# parser.add_argument('--no_eval_train', dest='eval_train', action='store_false')
# parser.set_defaults(eval_train=True)

# about loading and saving
parser.add_argument('--model_name_or_path', type=str, default='')
parser.add_argument('--output_model_dir', type=str, default='')

# verbosity
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no_verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=False)

parser.add_argument('--backbone',type=str,default='gnn')
parser.add_argument('--transform_in_collator', action='store_true')
parser.add_argument('--rich_features',action='store_true')

parser.add_argument('--return_model_size',action='store_true')


def get_num_task(dataset):
    """ used in molecule_finetune.py """
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp', 'donor','esol','freesolv','lipo']:
        return 1
    elif dataset == 'pcba':
        return 108
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    elif dataset == 'cyp450':
        return 5
    raise ValueError(dataset+': Invalid dataset name.')

def task_type(dataset):
    if dataset in ['esol','freesolv','lipo']:
        return 'reg'
    else:
        return 'cla'

def better_result(result,reference,dataset):
    if task_type(dataset)=='cla':
        return result>reference
    else:
        assert task_type(dataset)=='reg'
        return result<reference



args,left = parser.parse_known_args()

assert args.backbone in ['graphormer', 'gnn','kvplm','grapht5']
if args.backbone == 'graphormer':
    parsernew = HfArgumentParser(GraphormerConfig)
    # parsernew = argparse.ArgumentParser()
    parsernew = GraphormerModel.add_args(parsernew)
    graph_args = parsernew.parse_args(left)
    graph_args=Graphormer_version_dict[graph_args.arch](graph_args)
    # print('graphormer_args',graphormer_args)
elif args.backbone == 'gnn':
    parsernew = HfArgumentParser(GinConfig)
    graph_args = parsernew.parse_args(left)
elif args.backbone == 'kvplm':
    parsernew = HfArgumentParser(KVPLMConfig)
    graph_args = parsernew.parse_args(left)
elif args.backbone == 'grapht5':
    parsernew = HfArgumentParser(GraphormerConfig)
    # parsernew = argparse.ArgumentParser()
    parsernew = GraphormerModel.add_args(parsernew)
    graph_args = parsernew.parse_args(left)
    graph_args=Graphormer_version_dict[graph_args.arch](graph_args)
    # print('graphormer_args',graphormer_args)
    if task_type(args.dataset)=='cla':
        graph_args.graphonly_problem_type='multi_label_classification'
    else:
        graph_args.graphonly_problem_type = 'regression'
assert args.batch_size % args.grad_accum_step==0
args.batch_size_ori=args.batch_size
args.batch_size=args.batch_size_ori//args.grad_accum_step


print('arguments\t', args)


def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0
    # data_list = []
    # for data in tqdm(loader):
    #     data_list.append(data.edge_input.max())
    for step, batch in tqdm(enumerate(loader)):
        batch = batch.to(device)
        if args.backbone=='grapht5':
            pred = model(graph=batch,labels=batch.y)['logits']
        else:
            pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y != -100
        # Loss matrix
        loss_mat = criterion(pred.double(), y)
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))

        # optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss=loss/args.grad_accum_step
        loss.backward()
        # optimizer.step()
        if step % args.grad_accum_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.detach().item()

    return total_loss / len(loader)*args.grad_accum_step


def eval(model, device, loader):
    model.eval()
    y_true, y_scores = [], []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            if args.backbone!='grapht5':
                pred = model(batch)
            else:
                pred = model(batch)['logits']
    
        true = batch.y.view(pred.shape)

        y_true.append(true)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if task_type(args.dataset)=='cla':
        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                is_valid = y_true[:, i] !=-100
                roc_list.append(roc_auc_score(y_true[is_valid, i], y_scores[is_valid, i]))
            else:
                print('{} is invalid'.format(i))

        if len(roc_list) < y_true.shape[1]:
            print(len(roc_list))
            print('Some target is missing!')
            print('Missing ratio: %f' %(1 - float(len(roc_list)) / y_true.shape[1]))

        return sum(roc_list) / len(roc_list), 0, y_true, y_scores

    else:
        assert task_type(args.dataset)=='reg'
        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            # if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            ind = ~np.isnan(y_true[:, i])
            # true RMSE (was residual std in upstream GIMLET); lower is better
            mrs = float(np.sqrt(np.mean((y_true[ind, i] - y_scores[ind, i]) ** 2)))
            roc_list.append(mrs)
            # roc_list.append(r2_score(y_true[ind, i], y_scores[ind, i]))

            # # ratio=ind.float().mean()
            # # y_true=y_true[ind]
            # # y_scores=y_scores[ind]
            #
            # mrs=(y_true-y_scores).std()
            # naive_msr=(y_true-y_true.mean()).std()
            #
            # corrcoef=np.corrcoef(y_true,y_scores)[0,1]
            #
            # try:
            #     r2=r2_score(y_true,y_scores)
            # except:
            #     r2=np.nan


        if len(roc_list) < y_true.shape[1]:
            print(len(roc_list))
            print('Some target is missing!')
            print('Missing ratio: %f' %(1 - float(len(roc_list)) / y_true.shape[1]))



        return sum(roc_list) / len(roc_list), 0, y_true, y_scores



if __name__ == '__main__':
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device)) \
        if (not args.no_cuda) else torch.device('cpu')
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    args.num_tasks = get_num_task(args.dataset)
    dataset_folder = 'property_data/'



    if args.backbone == 'kvplm':
        dataset = MoleculeDatasetRich(root=dataset_folder,name=args.dataset, return_id=True,return_smiles=True,rich_features=args.rich_features)
    else:
        dataset = MoleculeDatasetRich(root=dataset_folder,name=args.dataset,rich_features=args.rich_features)
    print(dataset)





    if args.split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        # pos_ids = []
        # for i, data in enumerate(dataset):
        #     if int(data.y) == -1:
        #         pos_ids.append(i)
        # pos_smiles=[smiles_list[i] for i in pos_ids]
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1)
        print('split via scaffold')

        # --- MolE-RAG parity: override ALL THREE splits with our seed-0 splits (canonical-SMILES match) ---
        _ms = os.environ.get("MOLERAG_SPLITS_JSON")
        if _ms:
            from rdkit import Chem as _Chem
            def _cn(s):
                try:
                    m = _Chem.MolFromSmiles(s); return _Chem.MolToSmiles(m) if m else None
                except Exception:
                    return None
            _spl = _json.load(open(_ms))[args.dataset]
            _tr = set(x for x in (_cn(s) for s in _spl["train"]) if x)
            _va = set(x for x in (_cn(s) for s in _spl["valid"]) if x)
            _te = set(x for x in (_cn(s) for s in _spl["test"]) if x)
            _sl = [_cn(s) for s in smiles_list]
            tr_idx, va_idx, te_idx = [], [], []
            for i, c in enumerate(_sl):
                if c is None:
                    continue
                if c in _te:       # priority: protect test integrity
                    te_idx.append(i)
                elif c in _va:
                    va_idx.append(i)
                elif c in _tr:
                    tr_idx.append(i)
            train_dataset = dataset[torch.tensor(tr_idx, dtype=torch.long)]
            valid_dataset = dataset[torch.tensor(va_idx, dtype=torch.long)]
            test_dataset = dataset[torch.tensor(te_idx, dtype=torch.long)]
            print("[MOLERAG] %s split override: train=%d valid=%d test=%d (corpus %d; molerag sets tr/va/te=%d/%d/%d)"
                  % (args.dataset, len(tr_idx), len(va_idx), len(te_idx), len(_sl), len(_tr), len(_va), len(_te)), flush=True)
            assert len(te_idx) > 0 and len(tr_idx) > 0, "MolE-RAG split override produced empty train/test"
    elif args.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.runseed)
        print('randomly split')
    elif args.split == 'random_scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1, seed=args.runseed)
        print('random scaffold')
    else:
        raise ValueError('Invalid split option.')
    print(train_dataset[0])

    data_collator=GraphData_collator[args.backbone](transform_in_collator=args.transform_in_collator,include_y=True,rich_features=args.rich_features)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,collate_fn=data_collator)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,collate_fn=data_collator)

    # set up model
    model,optimizer=get_graph_model(args,graph_args)
    model.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.return_model_size:
        print('Model size: {}'.format(count_parameters(model)))


    # print(model)

    # set up optimizer
    # different learning rates for different parts of GNN

    criterion = nn.BCEWithLogitsLoss(reduction='none') if task_type(args.dataset)=='cla' else torch.nn.MSELoss(reduction='none')
    train_roc_list, val_roc_list, test_roc_list = [], [], []
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    best_val_roc, best_val_idx = None, 0

    for epoch in range(1, args.epochs + 1):
        loss_acc = train(model, device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        if args.eval_train:
            train_roc, train_acc, train_target, train_pred = eval(model, device, train_loader)
        else:
            train_roc = train_acc = 0
        val_roc, val_acc, val_target, val_pred = eval(model, device, val_loader)
        test_roc, test_acc, test_target, test_pred = eval(model, device, test_loader)

        train_roc_list.append(train_roc)
        train_acc_list.append(train_acc)
        val_roc_list.append(val_roc)
        val_acc_list.append(val_acc)
        test_roc_list.append(test_roc)
        test_acc_list.append(test_acc)
        print('train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc, val_roc, test_roc))
        print()

        if best_val_roc is None:
            best_val_roc=val_roc
            assert best_val_idx==0

        if better_result(val_roc, best_val_roc,args.dataset):
            best_val_roc = val_roc
            best_val_idx = epoch - 1
            if not args.output_model_dir == '':
                output_model_path = join(args.output_model_dir, 'model_best.pth')
                saved_model_dict = model.state_dict()
                torch.save(saved_model_dict, output_model_path)

                filename = join(args.output_model_dir, 'evaluation_best.pth')
                np.savez(filename, val_target=val_target, val_pred=val_pred,
                         test_target=test_target, test_pred=test_pred)

    # print('best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc_list[best_val_idx], val_roc_list[best_val_idx], test_roc_list[best_val_idx]))
    # with open('result.log', 'a+') as f:
    #     f.write(args.dataset + ' ' +args.input_model_file+ ' ' + str(args.runseed) + ' ' + 'best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc_list[best_val_idx], val_roc_list[best_val_idx], test_roc_list[best_val_idx]))
    #     f.write('\n')
    method_name=args.backbone if args.backbone!='gnn' else graph_args.gnn_type
    record = [(args.dataset,method_name,getattr(graph_args, "restore_file_graphormer", None), 'rich_features:'+str(args.rich_features), 'epochs:'+str(args.epochs), 'lr:'+str(args.lr), 'runseed:'+str(args.runseed), 'best_val_idx:'+str(best_val_idx),
               train_roc_list[best_val_idx], val_roc_list[best_val_idx], test_roc_list[best_val_idx])]
    df = pd.DataFrame(record,
                      columns=['dataset', 'backbone','input_model_file','rich_features', 'epoch', 'lr', 'runseed', 'best_val_idx', 'train_best',
                               'valid_best', 'test_best'
                               ])
    df.to_csv(join('cache','result_graph_transformer_graph_only.csv'), mode='a', header=False)

    # --- emit machine-readable per-run result for aggregation ---
    _rj = os.environ.get("MOLERAG_RESULT_JSON")
    if _rj:
        rec = {"dataset": args.dataset, "backbone": method_name, "runseed": int(args.runseed),
               "metric": ("RMSE" if task_type(args.dataset) == "reg" else "ROC-AUC"),
               "best_val_idx": int(best_val_idx), "epochs": int(args.epochs), "lr": float(args.lr),
               "val_best": float(val_roc_list[best_val_idx]),
               "test_best": float(test_roc_list[best_val_idx]),
               "n_test": int(len(test_dataset)), "n_train": int(len(train_dataset)),
               "restore_file": getattr(graph_args, "restore_file_graphormer", None)}
        with open(_rj, "a") as _f:
            _f.write(_json.dumps(rec) + "\n")
        print("[MOLERAG_RESULT]", _json.dumps(rec), flush=True)

    if args.output_model_dir is not '':
        output_model_path = join(args.output_model_dir, 'model_final.pth')
        saved_model_dict = model.state_dict()
        torch.save(saved_model_dict, output_model_path)
