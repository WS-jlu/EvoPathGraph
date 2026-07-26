import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np

import dgl
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from model import GCN, MLP, GAT, SAGE, GIN

from data_process import SurvivalDatasetFactory, make_weights
from loss import NLLSurvLoss, ce_loss
from utils import _summary, _update_arrays, _calculate_risk, DGLDataFrameDataset, eval_model
import argparse

from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score
from sksurv.util import Surv

import time
import random

from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.sampler import WeightedRandomSampler


from logger import Logger
from utils import args_print
from datetime import datetime

from copy import deepcopy

from sklearn.metrics import accuracy_score
import torch.nn.functional as F


warnings.filterwarnings("ignore") 


def seed_torch(seed=2027):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    seed_torch(args.seed)
    torch.cuda.set_device(args.device)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f'{args.dataset_name}_{args.model}_{args.task}__warm_up{args.warm_up}_{datetime_now}'
    exp_dir = os.path.join('./logs/', experiment_name)
    os.mkdir(exp_dir)
    logger = Logger.init_logger(filename=exp_dir + '/log.log')
    args_print(args, logger)

    start_time = time.time()
    dataset = args.dataset_name
    dataset_folder_path = f"{dataset}/homogeneous/"
    all_items = os.listdir(dataset_folder_path)
    data = pd.DataFrame(all_items, columns=["file_name"])
    index = data["file_name"].str.split("-").str[3].str[:2]
    data = data[index == '01']
    data["case_id"] = data["file_name"].str[:12]
    data = data.groupby("case_id", group_keys=False).first().reset_index()

    ############################################
    labeld_data = pd.read_csv(f"./primary_stats/{dataset}_primary_stats.csv")
    data = pd.merge(data, labeld_data, on='case_id', how='inner')
    if args.task == "Stage":
        final_mapping={'I': 0, 'IA': 0, 'IB': 0, 'II': 1, 'IIA': 1, 'IIB': 1, 'IIC': 1, 'III': 2, 'IIIA': 2, 'IIIB': 2, 'IIIC': 2, 'IV': 3, 'IVA': 3, 'IVB': 3, 'IVC': 3, 'X': 4}
        data["stage_value_map"]=data["stage_value"].map(final_mapping)
        counts = data["stage_value_map"].value_counts()
        rare_classes = counts[counts < 3].index.tolist()
        data = data[~data["stage_value_map"].isin(rare_classes)]

    elif args.task == "OS" or args.task == "PFS":
        data["os_value"] = data["os_value"] / 30.
        data["pfs_value"] = data["pfs_value"] / 30.
        os_sdf = SurvivalDatasetFactory(label_file=data[["case_id", "os_censorship", "os_value"]].copy(),
                                        n_bins=args.n_classes, label_col="os_value", censorship_var="os_censorship")
        os_disc_labels, os_q_bins = os_sdf.get_target()
        pfs_sdf = SurvivalDatasetFactory(label_file=data[["case_id", "pfs_censorship", "pfs_value"]].copy(),
                                        n_bins=args.n_classes, label_col="pfs_value", censorship_var="pfs_censorship")
        pfs_disc_labels, pfs_q_bins = pfs_sdf.get_target()
        data["os_disc_labels"] = os_disc_labels
        data["pfs_disc_labels"] = pfs_disc_labels
    else:
        print("Wrong task!")
    ############################################

    
    best_weights = None
    all_info=[]
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    if args.task == "OS":
        kflod_split_data=skf.split(data,data['os_censorship'])
    elif args.task == "PFS":
        kflod_split_data=skf.split(data,data['pfs_censorship'])
    elif args.task == "Stage":
        kflod_split_data=skf.split(data,data['stage_value_map'])

    for fold, (train_val_idx, test_idx) in enumerate(kflod_split_data, 1):
        train_val_data=data.iloc[train_val_idx]
        test_data=data.iloc[test_idx]

        if args.task == "OS":
            train_data, val_data =train_test_split(train_val_data, test_size=1/8, stratify=train_val_data['os_censorship'], random_state=args.seed) 
            all_data_censorships = data['os_censorship'].to_numpy()
            all_data_event_times = data['os_value'].to_numpy()
            all_survival = Surv.from_arrays(event=(1 - all_data_censorships).astype(bool), time=all_data_event_times)
            test_event_times = test_data['os_value']
            val_event_times =val_data['os_value']
            bins = os_q_bins
        elif args.task == "PFS":
            train_data, val_data =train_test_split(train_val_data, test_size=1/8, stratify=train_val_data['pfs_censorship'], random_state=args.seed)
            all_data_censorships = data['pfs_censorship'].to_numpy()
            all_data_event_times = data['pfs_value'].to_numpy()
            all_survival = Surv.from_arrays(event=(1 - all_data_censorships).astype(bool), time=all_data_event_times)
            test_event_times = test_data['pfs_value']
            val_event_times =val_data['pfs_value']
            bins = pfs_q_bins
        elif args.task == "Stage":
            train_data, val_data =train_test_split(train_val_data, test_size=1/8, stratify=train_val_data['stage_value_map'], random_state=args.seed)



        logger.info(f"# All: {len(data)} # Train: {len(train_data)}  #Val: {len(val_data)} #Test: {len(test_data)} ")
        attribute_cols = ['case_id']
        train_weighted_sampler = WeightedRandomSampler(weights=torch.DoubleTensor(make_weights(args, train_data.copy())),
                                                    num_samples=train_data.shape[0], replacement=True)

        train_dataset = DGLDataFrameDataset(train_data, args.task, graph_attribute_cols=attribute_cols, data_root=dataset)
        train_dataloader = dgl.dataloading.GraphDataLoader(train_dataset, sampler=train_weighted_sampler, batch_size=args.batch_size, drop_last=False)
        val_dataset = DGLDataFrameDataset(val_data, args.task, graph_attribute_cols=attribute_cols, data_root=dataset)
        val_dataloader = dgl.dataloading.GraphDataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,  drop_last=False)
        test_dataset = DGLDataFrameDataset( test_data, args.task, graph_attribute_cols=attribute_cols, data_root=dataset)
        test_dataloader = dgl.dataloading.GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,  drop_last=False)

        logger.info(f"########################## model:{args.model} dataset:{args.dataset_name} seed:{args.seed} fold:{fold} ###################")
        if args.model.lower() == "gcn":
            model = GCN(args.input_dim, args.hidden, args.n_classes, args.dropout_rate).to(args.device)
        elif args.model.lower() == "mlp":
            model = MLP(args.input_dim, args.hidden, args.n_classes, args.dropout_rate).to(args.device)
        elif args.model == "gat":
            model = GAT(args.input_dim, args.hidden, args.n_classes, args.dropout_rate).to(args.device) 
        elif args.model.lower() == "sage":  
            model = SAGE(args.input_dim, args.hidden, args.n_classes, args.dropout_rate, args.n_layers).to(args.device)     
        elif args.model == "gin":
            model = GIN(args.input_dim, args.hidden, args.n_classes, args.dropout_rate, args.n_layers).to(args.device)       
        elif args.model.lower() == "gcnmlp":
            model = GCNMLP(args.input_dim, args.hidden, args.n_classes, args.dropout_rate).to(args.device)  

        else:
            print("No model")  

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.task=="Stage":
            loss_fn = ce_loss
        else:
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        best_val_eval =  0
        cnt = 0# early stopping count
        best_val_test=[]


        for epoch in range(args.max_epochs):
            model.train()
            total_loss = 0.
            if args.task == "Stage":
                all_probs = []
                all_labels = []
                all_preds = []
            elif args.task == "OS" or args.task == "PFS":
                all_risk_scores = []
                all_censorships = []
                all_event_times = []
            else:
                print("Wrong Task!")
            for batch, label, attribute in train_dataloader:
                optimizer.zero_grad()
                if args.task == "Stage":
                    stage_value_map = label[0]
                    g = batch.to(args.device)
                    feats = g.ndata['feat']
                    logits = model(g, feats)
                    loss = loss_fn(logits, stage_value_map)
                    loss_value = loss.item()                   
                elif args.task == "OS" or args.task == "PFS":
                    os_value, pfs_value, os_disc_labels, pfs_disc_labels, stage_value, os_censorship, pfs_censorship = label
                    g = batch.to(args.device)
                    feats = g.ndata['feat']
                    logits = model(g, feats)
                    if args.task == "OS":
                        loss = loss_fn(h=logits, y=os_disc_labels, c=os_censorship)
                        loss_value = loss.item()
                        loss = loss / os_disc_labels.shape[0]
                    elif args.task == "PFS":
                        loss = loss_fn(h=logits, y=pfs_disc_labels, c=pfs_censorship)
                        loss_value = loss.item()
                        loss = loss / pfs_disc_labels.shape[0]
                    risk, _ = _calculate_risk(h=logits)
                else:
                    print("Wrong Task!")


                if args.task == "OS":
                    all_risk_scores, all_censorships, all_event_times = _update_arrays(all_risk_scores, all_censorships,
                                                                                    all_event_times, os_value,
                                                                                    os_censorship,
                                                                                    risk)
                elif args.task == "PFS":
                    all_risk_scores, all_censorships, all_event_times = _update_arrays(all_risk_scores, all_censorships,
                                                                                    all_event_times, pfs_value,
                                                                                    pfs_censorship,
                                                                                    risk)
                elif args.task == "Stage":
                    probs = F.softmax(logits, dim=1)
                    _, preds = torch.max(probs, 1)
                    all_probs.append(probs.detach().cpu().numpy())
                    all_preds.append(preds.detach().cpu().numpy())#
                    all_labels.append(stage_value_map.detach().cpu().numpy())

                total_loss += loss_value
                loss.backward()
                optimizer.step()
            # total_loss 
            if args.task == "Stage":
                all_preds = np.concatenate(all_preds, axis=0)
                all_probs = np.concatenate(all_probs, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                acc = accuracy_score(all_labels, all_preds)
                val_acc, val_macro, val_micro, val_total_loss = eval_model(args, model, val_dataloader, loss_fn)
                test_acc, test_macro, test_micro, test_total_loss = eval_model(args, model, test_dataloader, loss_fn)
                evaluate=val_acc
                logger.info('Epoch: {}, train_loss: {:.4f}, train_acc: {:.4f},  val_evaluate: {:.4f}'.format(epoch, total_loss, acc, evaluate))
                if epoch >= args.warm_up:
                    if  evaluate > best_val_eval:
                        best_val_eval = evaluate
                        best_val_test=[test_acc, test_macro, test_micro]
                        if args.save_model:
                            best_weights = deepcopy(model.state_dict())
                        cnt = 0
                    else:
                        cnt = cnt +1

                if epoch >= args.warm_up and cnt >= args.early_stopping:
                    logger.info("Early Stopping")
                    logger.info("+" * 50)
                    logger.info(
                        'Last Test acc: {:.4f} |macro: {:.4f} |micro: {:.4f}'.format(
                            best_val_test[0],
                            best_val_test[1],
                            best_val_test[2]
                        ))   
                    logger.info("+" * 50)               
                    break            
      
            elif args.task == "OS" or args.task == "PFS":
                all_risk_scores = np.concatenate(all_risk_scores, axis=0)
                all_censorships = np.concatenate(all_censorships, axis=0)
                all_event_times = np.concatenate(all_event_times, axis=0)
                c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                                    tied_tol=1e-08)[0]

                
                val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_total_loss = _summary(args, model,  val_dataloader, loss_fn, all_survival, bins, val_event_times)
                test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_total_loss = _summary(args, model,  test_dataloader, loss_fn, all_survival, bins, test_event_times)
                evaluate=val_cindex
                logger.info('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f},  val_evaluate: {:.4f}'.format(epoch, total_loss, c_index, evaluate))
                if epoch >= args.warm_up:
                    if  evaluate > best_val_eval:
                        best_val_eval = evaluate
                        best_val_test=[test_cindex, test_cindex_ipcw, test_BS, test_IBS]
                        if args.save_model:
                            best_weights = deepcopy(model.state_dict())
                        cnt = 0
                    else:
                        cnt = cnt +1

                if epoch >= args.warm_up and cnt >= args.early_stopping:
                    logger.info("Early Stopping")
                    logger.info("+" * 50)
                    logger.info(
                        'Last Test c-index: {:.4f} |c-index2: {:.4f} |BS: {:.4f} |IBS: {:.4f}'.format(
                            best_val_test[0],
                            best_val_test[1],
                            best_val_test[2],
                            best_val_test[3]
                        ))   
                    logger.info("+" * 50)               
                    break                     
            else:
                print("Wrong Task!")


        all_info.append(best_val_test)
        save_name = f'{args.dataset_name}_{datetime_now}_{args.seed}'
        if args.save_model:
            logger.info("Saving best weights..")
            model_path = os.path.join('save_model', save_name) + ".pt"
            for k, v in best_weights.items():
                best_weights[k] = v.cpu()
            torch.save(best_weights, model_path)
            logger.info("Done..")  

    all_info=torch.tensor(all_info)
    all_info_mean=all_info.mean(dim=0)
    all_info_std=all_info.std(dim=0)
    logger.info(all_info)
    logger.info(all_info_mean)
    logger.info(all_info_std)
    if args.task == "Stage":
        logger.info('Finall Test acc: {:.4f}-+-{:.4f}|macro: {:.4f}-+-{:.4f} |micro: {:.4f}-+-{:.4f}'.format(
                all_info_mean[0],
                all_info_std[0],
                all_info_mean[1],
                all_info_std[1],
                all_info_mean[2],
                all_info_std[2],

            ))
    
    else:
        logger.info('Finall Test c-index: {:.4f}-+-{:.4f}|c-index2:{:.4f}-+-{:.4f} |BS: {:.4f}-+-{:.4f} |IBS: {:.4f}-+-{:.4f}'.format(
                all_info_mean[0],
                all_info_std[0],
                all_info_mean[1],
                all_info_std[1],
                all_info_mean[2],
                all_info_std[2],
                all_info_mean[3],
                all_info_std[3]
            ))
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Program Execution Time: {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='yiliao')
    parser.add_argument('--seed', type=int, default=2026, help='the seed used in the training')
    parser.add_argument('--dataset_name', type=str, default="blca", help='[coad, brca, clca, hsne, stad]')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate for the predictor')
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)')
    parser.add_argument('--task', type=str, default="OS", help='type of survival (OS, PFI, Stage)')
    parser.add_argument('--alpha_surv', type=float, default=0.5, help='weight given to uncensored patients')
    parser.add_argument("--input_dim", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument('--model', default='mlp', type=str, help='[mlp, gcn, gat, sage, gin]')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='maximum number of epochs to train (default: 20)')  # 100
    parser.add_argument('--batch_size', type=int, default=32)  #
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--save_model', action='store_true') 
    parser.add_argument('--warm_up', default=2, type=int)
    parser.add_argument('--early_stopping', default=5, type=int)
    parser.add_argument('--weight_decay', default=5e-6, type=float) 
    parser.add_argument('--dropout_rate', default=0.5, type=float) 
    parser.add_argument('--n_layers', default=2, type=int) 
    parser.add_argument('--kfold', default=5, type=int) 
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    main(args)





