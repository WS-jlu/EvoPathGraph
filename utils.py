import torch
import torch.nn.functional as F
import numpy as np
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score
from sksurv.util import Surv
from dgl.data import DGLDataset
from dgl import DGLGraph
import dgl
import pickle
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score



class DGLDataFrameDataset(DGLDataset):
    def __init__(self, df: pd.DataFrame, task: str = "OS", graph_attribute_cols: list = None, data_root: str = "./"):
        super(DGLDataFrameDataset, self).__init__(name='graph_classification_dataset')
        self.df = df.reset_index(drop=True)
        self.graph_attribute_cols = graph_attribute_cols
        self.file_paths = self.df['file_name'].tolist()
        self.data_root = data_root
        self.task = task

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        graph = self.__getgraph__(idx)
        # graph.ndata['node_type']=self.__getnode_type_info__(idx)
        row = self.df.iloc[idx]
        attribute = tuple(row[self.graph_attribute_cols].values) 
        if self.task == "Stage":
            labels = tuple([row['stage_value_map']]) 
        elif self.task == "CancerCls":
            labels = tuple([row['cancer_label']]) 
        else: 
            labels = tuple([
                row['os_value'],
                row['pfs_value'],
                row['os_disc_labels'],
                row['pfs_disc_labels'],
                row['stage_value'], row['os_censorship'], row['pfs_censorship']]
            ) 
        return graph, labels, attribute

    def __getgraph__(self, idx):
        file_path = self.data_root + "/homogeneous/" + self.file_paths[idx]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            graph = pickle.load(f)
            graph = dgl.add_self_loop(graph)

        if not isinstance(graph, DGLGraph):
            raise TypeError(f"Loaded object from {file_path} is not a DGLGraph.")
        return graph





def eval_model(args, model, test_dataloader, loss_fn):
    model.eval()
    total_loss = 0.
    all_probs = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch, label, attribute in test_dataloader:
            stage_value_map = label[0]
            g = batch.to(args.device)
            feats = g.ndata['feat']
            h = model(g, feats)
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            loss = loss_fn(h,stage_value_map)
            loss_value = loss.item()
            probs = F.softmax(h, dim=1)
            _, preds = torch.max(probs, 1)
            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())#
            all_labels.append(stage_value_map.detach().cpu().numpy())
            total_loss += loss_value
      
    total_loss /= len(test_dataloader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    acc = accuracy_score(all_labels, all_preds)
    macro = f1_score(all_labels, all_preds, average='macro')
    micro = f1_score(all_labels, all_preds, average='micro')

    return acc, macro, micro, total_loss





def _summary(args, model, test_dataloader, loss_fn, all_survival, bins, test_event_times):


    task = args.task


    model.eval()

    total_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_logits = []

    with torch.no_grad():
        for batch, label, attribute in test_dataloader:
            os_value, pfs_value, os_disc_labels, pfs_disc_labels, stage_value, os_censorship, pfs_censorship= label
            g = batch.to(args.device)
            feats = g.ndata['feat']
            h = model(g, feats)

            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            if task == "OS":
                loss = loss_fn(h=h, y=os_disc_labels, c=os_censorship)
                event_time = os_value
                censor=os_censorship
            elif task == "PFS":
                loss = loss_fn(h=h, y=pfs_disc_labels, c=pfs_censorship)
                event_time = pfs_value
                censor=pfs_censorship
            loss_value = loss.item()
            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times = _update_arrays(all_risk_scores,all_censorships, all_event_times,event_time, censor,risk)

            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
    

    total_loss /= len(test_dataloader.dataset)
    test_all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    test_all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    test_all_censorships = np.concatenate(all_censorships, axis=0)
    test_all_event_times = np.concatenate(all_event_times, axis=0)


    c_index, c_index2, BS, IBS  = _calculate_metrics( bins, test_event_times, all_survival, test_all_risk_scores,
                                                          test_all_censorships, test_all_event_times, test_all_risk_by_bin_scores)

    return c_index, c_index2, BS, IBS, total_loss


def _calculate_metrics( bins, test_event_times, all_survival, test_all_risk_scores, test_all_censorships, test_all_event_times,
                       test_all_risk_by_bin_scores):


    data = test_event_times
    which_times_to_eval_at = np.array([bins[i] for i in range(len(bins)-1)])
    which_times_to_eval_at[0]=data.min() + 0.0001
    which_times_to_eval_at[-1]=data.max() - 0.0001

    original_risk_scores = test_all_risk_scores
    test_all_risk_scores = np.delete(test_all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    test_all_censorships = np.delete(test_all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    test_all_event_times = np.delete(test_all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    # <---

    c_index = \
    concordance_index_censored((1 - test_all_censorships).astype(bool), test_all_event_times, test_all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS = 0., 0., 0.

    try:
        survival_test = Surv.from_arrays(event=(1 - test_all_censorships).astype(bool), time=test_all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS

    try:
        c_index_ipcw = concordance_index_ipcw(all_survival, survival_test, estimate=test_all_risk_scores)[0]

    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.

    # brier score
    try:
        _, BS = brier_score(all_survival, survival_test, estimate=test_all_risk_by_bin_scores,
                            times=which_times_to_eval_at)
        BS=np.mean(BS)
    except:
        print('An error occured while computing BS')
        BS = 0.

    # IBS
    try:
        IBS = integrated_brier_score(all_survival, survival_test, estimate=test_all_risk_by_bin_scores,
                                     times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.



    return c_index, c_index_ipcw, BS, IBS


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient

    Args:
        - h : torch.Tensor

    Returns:
        - risk : torch.Tensor

    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()


def _update_arrays(all_risk_scores, all_censorships, all_event_times, event_time, censor, risk):
    r"""
    Update the arrays with new values

    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List

    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List

    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    return all_risk_scores, all_censorships, all_event_times#, all_clinical_data





def args_print(args, logger):
    logger.info("=" * 50)
    _dict = vars(args)
    logger.info("Parameter : Value")
    for k, v in _dict.items():
        logger.info(f"{k}: {v}")
    logger.info("=" * 50)