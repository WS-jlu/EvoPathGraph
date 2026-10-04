import os
import sys
import argparse
import pickle
import glob


from itertools import chain
import nmslib
import dgl
from scipy.stats import pearsonr

import torch
from pathlib import Path
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import numpy as np
import torch.nn as nn  


import warnings
warnings.filterwarnings("ignore")



class fully_connected(nn.Module):
    """docstring for BottleNeck"""

    def __init__(self, model, num_ftrs, num_classes):
        super(fully_connected, self).__init__()
        self.model = model
        self.fc_4 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        out_1 = x
        out_3 = self.fc_4(x)
        return out_1, out_3




class KimiaNet_infer:
    def __init__(self, config, device):
   

        # self.model = torchvision.models.densenet121(pretrained=True)
        self.model = torchvision.models.densenet121(pretrained=False)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.features = nn.Sequential(self.model.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        num_ftrs = self.model.classifier.in_features
        self.model_final = fully_connected(self.model.features, num_ftrs, 30)
        self.device = device
        self.model_final = nn.DataParallel(self.model_final)
        self.model_final = self.model_final.to(self.device)
        # self.model_final = nn.DataParallel(self.model_final)

        state_dict = torch.load(config)

        sd = self.model_final.state_dict()
        for (k, v), ky in zip(state_dict.items(), sd.keys()):
            sd[ky] = v
        self.model_final.load_state_dict(sd)

    def predict(self, dataloader):
        self.model_final.eval()
        features_list = []
        for idx, data in enumerate(dataloader):
            # data = data.permute(0, 3, 2, 1)
            data = data.to(self.device)
            output1, _ = self.model_final(data)
            output_1024 = output1.cpu().detach().numpy()
            features_list.append(output_1024)
        return np.concatenate(features_list)





class Hnsw:
    """
    KNN model cloned from https://github.com/mahmoodlab/Patch-GCN/blob/master/WSI-Graph%20Construction.ipynb
    """

    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices



def construct_graph(features, radius):


    # Number of patches
    n_patches = features.shape[0]
    knn_model = Hnsw(space='l2')
    # Construct graph using spatial coordinates
    knn_model.fit(features)

    a = np.repeat(range(n_patches), radius - 1)
    b = np.fromiter(
        chain(
            *[knn_model.query(features[v_idx], topn=radius)[1:] for v_idx in range(n_patches)]
        ), dtype=int
    )
    edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)


    features = torch.tensor(features, device='cpu').float()
    homo_graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
    homo_graph.ndata['feat'] = features
   
    return homo_graph, edge_spatial





def construct_spatial_graph(features, patch_list, adjacency):

    a,b=build_spatial_edges(patch_list, adjacency)
    edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)
    features = torch.tensor(features, device='cpu').float()
    homo_graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
    homo_graph.ndata['feat'] = features
    return homo_graph, edge_spatial



def construct_mix_graph(features, radius, patch_list, adjacency):
 
    n_patches = features.shape[0]
    knn_model = Hnsw(space='l2')
    # Construct graph using spatial coordinates
    knn_model.fit(features)

    a_knn = np.repeat(range(n_patches), radius - 1)
    b_knn = np.fromiter(
        chain(
            *[knn_model.query(features[v_idx], topn=radius)[1:] for v_idx in range(n_patches)]
        ), dtype=int
    )
    edge_knn = torch.Tensor(np.stack([a_knn, a_knn])).type(torch.LongTensor)
    a_spatial,b_spatial=build_spatial_edges(patch_list, adjacency)
    edge_spatial = torch.Tensor(np.stack([a_spatial, b_spatial])).type(torch.LongTensor)

    edge_mix = torch.cat([edge_spatial, edge_knn], dim=1)
    features = torch.tensor(features, device='cpu').float()
    homo_graph = dgl.graph((edge_mix[0, :], edge_mix[1, :]))
    homo_graph.ndata['feat'] = features
    return homo_graph, edge_mix

def mix_graph(features, edge_knn, edge_spatial):
    edge_mix = torch.cat([edge_spatial, edge_knn], dim=1)
    features = torch.tensor(features, device='cpu').float()
    homo_graph = dgl.graph((edge_mix[0, :], edge_mix[1, :]))
    homo_graph.ndata['feat'] = features
    return homo_graph, edge_mix



def build_spatial_edges(patch_list, adjacency):

    coords = [tuple(map(int, name.split('_'))) for name in patch_list]

    coord_to_idx = {c: i for i, c in enumerate(coords)}

    if adjacency == '4nn':
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif adjacency == '8nn':
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   ( 0, -1),          ( 0, 1),
                   ( 1, -1), ( 1, 0), ( 1, 1)]
    else:
        raise ValueError("no adjacency")

    src, dst = [], []
    for i, (r, c) in enumerate(coords):
        for dr, dc in offsets:
            j = coord_to_idx.get((r + dr, c + dc), -1)
            if j != -1:
                src.append(i)
                dst.append(j)

    return np.array(src), np.array(dst)






class PatchData(Dataset):
    def __init__(self, wsi_path):
        """
        Args:
            data_24: path to input data
        """
        self.patch_paths = [p for p in wsi_path.glob("*")]
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor(),

        ])

    def __len__(self):
        return len(self.patch_paths)
    
    def get_patch_name(self):
        patch_names = [p.stem  for p in self.patch_paths]
        return patch_names

    def __getitem__(self, idx):

        img = Image.open(self.patch_paths[idx]).convert('RGB')
        img = self.transforms(img)
        return img



def save_graph(save_dir,homo_graph):
    graph_output_file = os.path.join(save_dir + '/homogeneous/' + tail+ '.pkl')

    # Make directory
    if not Path(save_dir + '/homogeneous/').exists():
        Path(save_dir + '/homogeneous/').mkdir(parents=True)    
    #save        
    with open(graph_output_file, 'wb') as f:
        pickle.dump(homo_graph, f)
    print("Homo Graph saved at: " + graph_output_file)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K=9
    patch_path="./patches"
    patch_paths = glob.glob( patch_path + "*/*/*")
    model_name="kimianet"
    construct_list=['knn','4nn','knn4nn']

    #model

    if model_name == "kimianet":
        print("Use KimiaNet pretrained model!")
        kimia_model = KimiaNet_infer("checkpoints/kimianet/KimiaNetPyTorchWeights.pth", device)
        
    else:
        print("No model")

    for i, wsi_input in enumerate(patch_paths):
        print(f"Processing {i+1} / {len(patch_paths)}")
        print(wsi_input)
        head, tail = os.path.split(wsi_input)

        patch_path = Path(wsi_input)
        if model_name == "kimianet":
            patch_dataset = PatchData(patch_path)
            patch_list = patch_dataset.get_patch_name()
        else:
            print("No model")
        dataloader = data.DataLoader(
            patch_dataset,
            num_workers=0,
            batch_size=256,
            shuffle=False
        )
        g_emd = kimia_model.predict(dataloader)
        # print("g_emd", g_emd.shape)

        if 'knn' in construct_list:
            homo_graph, edge_knn = construct_graph(g_emd, K)#construct_graph(X, K), K is number of neighbors
            save_dir='./knn'
            save_graph(save_dir, homo_graph)
        
        if '4nn' in construct_list:
            homo_graph, edge_spatial = construct_spatial_graph(g_emd, patch_list, '4nn')
            save_dir='./4nn'
            save_graph(save_dir, homo_graph)

        if '8nn' in construct_list:
            homo_graph, edge_spatial = construct_spatial_graph(g_emd, patch_list, '8nn')
            save_dir='./8nn'
            save_graph(save_dir, homo_graph)


        if 'knn4nn' in construct_list:
            if edge_spatial is not None and edge_knn is not None:
                homo_graph, edge_mix = mix_graph(g_emd, edge_knn, edge_spatial)
            else:
                print("Without edge_spatial and edge_knn.")
                homo_graph, edge_mix = construct_mix_graph(g_emd, K, patch_list, '4nn')
            save_dir='./knn4nn'
            save_graph(save_dir, homo_graph)


        if 'knn8nn' in construct_list:
            if edge_spatial is not None and edge_knn is not None:
                print("edge_spatial and edge_knn", edge_spatial, edge_knn)
                homo_graph, edge_mix = mix_graph(g_emd, edge_knn, edge_spatial)
            else:
                print("Without edge_spatial and edge_knn.")
                homo_graph, edge_mix = construct_mix_graph(g_emd, K, patch_list, '8nn')
            save_dir='./knn8nn'
            save_graph(save_dir, homo_graph)


if __name__ == '__main__':
    main()

