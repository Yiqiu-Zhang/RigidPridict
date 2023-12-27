import os, pickle, copy
from abc import ABC

import torch
import torch.nn as nn
import torch_geometric
from rigid_predict.utlis.structure_build import get_gt_init_frames
from rigid_predict.utlis.constant import restype_frame_mask, middle_atom_mask
from torch_geometric.data import Dataset

def relpos(rigid_res_index, edge_index):

    d_i = rigid_res_index[edge_index[0]]
    d_j = rigid_res_index[edge_index[1]]

    # [E]
    d = d_i - d_j

    boundaries = torch.arange(start=-32, end=32 + 1, device=d.device)
    reshaped_bins = boundaries.view(1, len(boundaries))

    d = d[..., None] - reshaped_bins
    d = torch.abs(d)
    d = torch.argmin(d, dim=-1)

    return d

def rbf(D, D_min=0., D_max=20., D_count=16):
    # Distance radial basis function

    D_mu = torch.linspace(D_min, D_max, D_count).to(D)
    D_mu = D_mu.view([1] * len(D.shape) + [-1])

    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

    return RBF

def knn_graph(x, k):

    displacement = x[None, :, :] - x[:, None, :]
    distance = torch.linalg.vector_norm(displacement, dim=-1).float().to(x)

    # Value of distance [N_rigid, K], Index of distance [N_rigid, K]
    distance, E_idx = torch.topk(distance, k, dim=-1, largest=False)
    col = E_idx.flatten() # source
    row = torch.arange(E_idx.size(0)).view(-1,1).repeat(1,k).flatten().to(col) # target

    return torch.stack([row, col], dim=0), distance.flatten()

def update_edges(rigids, seq, rigid_mask, k=60,):

    res_index = torch.arange(0, len(seq))
    edge_index, distance = knn_graph(rigids.loc, k)

    distance_rbf = rbf(distance)
    rigid_res_idx = res_index.unsqueeze(-1).repeat(1, 5).reshape(-1)
    rigid_res_idx = rigid_res_idx[rigid_mask]

    relative_pos = relpos(rigid_res_idx, edge_index)

    return distance_rbf, relative_pos, edge_index

def protein_to_graph(protein):
    angles = torch.as_tensor(protein['angles'])
    seq = torch.as_tensor(protein['seq'])
    coords = torch.as_tensor(protein['coords'])
    chi_mask = torch.as_tensor(protein['chi_mask'])
    rigid_type_onehot = torch.as_tensor(protein['rigid_type_onehot'])
    rigid_property = torch.as_tensor(protein['rigid_property'])
    esm_s = torch.as_tensor(protein['acid_embedding'])
    fname = protein['fname']

    # rigid_mask
    restype_frame5_mask = torch.tensor(restype_frame_mask, dtype=bool)
    frame_mask = restype_frame5_mask[seq, ...]
    rigid_mask = torch.BoolTensor(torch.flatten(frame_mask, start_dim=-2))

    mid_frame_mask = torch.tensor(middle_atom_mask, dtype=bool)
    mid_frame_mask = mid_frame_mask[seq, ...]
    mid_frame_mask = torch.BoolTensor(torch.flatten(mid_frame_mask, start_dim=-2))
    mid_frame_mask = mid_frame_mask[rigid_mask]

    bb_mask = torch.zeros(*seq.shape,5, dtype=bool)
    bb_mask[:,0] = True
    bb_mask = torch.BoolTensor(torch.flatten(bb_mask, start_dim=-2))
    bb_mask = bb_mask[rigid_mask]

    flat_rigid_type = rigid_type_onehot.reshape(-1, rigid_type_onehot.shape[-1])
    flat_rigid_property = rigid_property.reshape(-1, rigid_property.shape[-1])
    # expand_seq = esm_s.repeat(1, 5).reshape(-1, esm_s.shape[-1])
    # [N_rigid, nf_dim] 7 + 19
    node_feature = torch.cat((flat_rigid_type, flat_rigid_property), dim=-1).float()
    node_feature = node_feature[rigid_mask]

    gt_rigids, local_r, gt_frames_to_global, init_rigid = get_gt_init_frames(angles, coords, seq, rigid_mask)

    k = 32 if len(init_rigid) >= 32 else len(init_rigid)
    distance_rbf, relative_pos, edge_index = update_edges(init_rigid, seq, rigid_mask, k)

    data = torch_geometric.data.Data(x = node_feature,
                                     esm_s = esm_s,
                                     true_chi=angles,
                                     aatype=seq,
                                     bb_coord=coords,
                                     chi_mask = chi_mask,
                                     rigid_mask = rigid_mask,
                                     fname = fname,
                                     edge_index = edge_index,
                                     atom_mask = mid_frame_mask,
                                     bb_mask = bb_mask,
                                     gt_rigids = gt_rigids,
                                     )

    return data


def preprocess_datapoints(graph_data=None, raw_dir=None):
    if graph_data and os.path.exists(graph_data):
        print('Reusing graph_data', graph_data)
        with open(graph_data, "rb") as f:
            proteins = pickle.load(f)
    else:
        print("Preprocessing")
        with open(raw_dir, "rb") as file:
            proteins_list = pickle.load(file)

        proteins = []
        for protein in proteins_list:
            proteins.append(protein_to_graph(protein))

        if graph_data:
            print("Store_data at", graph_data)
            with open(graph_data, "wb") as f:
                pickle.dump(proteins, f)
                print('finish data preprocess')

    return proteins

class ProteinDataset(Dataset, ABC):
    def __init__(self,data, transform = None):

        super(ProteinDataset, self).__init__(transform)
        self.transform = transform
        self.proteins = data

    def len(self): return len(self.proteins)
    def get(self, item):
        protein =  self.proteins[item]
        return copy.deepcopy(protein)