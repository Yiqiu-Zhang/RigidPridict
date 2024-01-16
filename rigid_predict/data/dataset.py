import os, pickle, copy
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from rigid_predict.utlis.structure_build import get_gt_init_frames, make_atom14_positions, frame_to_14pos
from torch_geometric.data import Dataset
from rigid_predict.utlis.geometry import from_tensor_4x4
from rigid_predict.data import data_transform
from rigid_predict.utlis import constant_test
from rigid_predict.utlis import structure_build
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
    col = E_idx.flatten()  # source
    row = torch.arange(E_idx.size(0)).view(-1, 1).repeat(1, k).flatten().to(col)  # target

    return torch.stack([row, col], dim=0), distance.flatten()


def update_edges(rigids, seq, rigid_mask, k=30,):
    res_index = torch.arange(0, len(seq))
    rigids = from_tensor_4x4(rigids)
    edge_index, distance = knn_graph(rigids.trans, k)

    distance_rbf = rbf(distance)
    rigid_res_idx = res_index.unsqueeze(-1).repeat(1, 3).reshape(-1)
    rigid_res_idx = rigid_res_idx[rigid_mask]

    relative_pos = relpos(rigid_res_idx, edge_index)

    return distance_rbf, relative_pos, edge_index


def protein_to_graph(protein):
    angles = torch.as_tensor(protein['angles'])
    seq = torch.as_tensor(protein['seq'])
    coords = torch.as_tensor(protein['coords'])
    chi_mask = torch.as_tensor(protein['chi_mask'])
    # rigid_type_onehot = torch.as_tensor(protein['rigid_type_onehot'])
    # rigid_property = torch.as_tensor(protein['rigid_property'])
    esm_s = torch.as_tensor(protein['acid_embedding'])
    fname = protein['fname']
    all_atom_positions = torch.as_tensor(protein["all_atom_positions"])

    gt_frames_tensor = data_transform.atom37_to_frames(seq,
                                                       all_atom_positions,
                                                       )
    # rigid_mask
    restype_frame5_mask = torch.tensor(constant_test.frame_mask, dtype=torch.bool)
    frame_mask = restype_frame5_mask[seq, ...]
    rigid_mask = torch.BoolTensor(torch.flatten(frame_mask, start_dim=-2))

    atom_mask = torch.tensor(constant_test.middle_atom_mask, dtype=torch.bool)
    atom_mask = atom_mask[seq, ...]
    atom_mask = torch.BoolTensor(torch.flatten(atom_mask, start_dim=-2))
    atom_mask = atom_mask[rigid_mask]

    bb_mask = torch.zeros(*seq.shape, 3, dtype=torch.bool)
    bb_mask[:, 0] = True
    bb_mask = torch.BoolTensor(torch.flatten(bb_mask, start_dim=-2))
    bb_mask = bb_mask[rigid_mask]

    restype_rigid_type = torch.tensor(constant_test.restype_rigidtype, dtype=torch.long)
    residx_rigid_type = restype_rigid_type[seq]
    restype_rigid_mask = torch.tensor(constant_test.restype_rigid_mask, dtype=torch.bool)
    residx_rigid_mask = restype_rigid_mask[seq]
    residx_rigid_type_onehot = F.one_hot(residx_rigid_type, 20) * residx_rigid_mask.unsqueeze(-1)

    flat_rigid_type = residx_rigid_type_onehot.reshape(-1, 20)
    # flat_rigid_property = rigid_property.reshape(-1, rigid_property.shape[-1])

    # expand_seq = esm_s.repeat(1, 5).reshape(-1, esm_s.shape[-1])
    # [N_rigid, nf_dim] 7 + 19
    node_feature = flat_rigid_type.float()[rigid_mask]

    init_rigid = structure_build.get_init_frames(gt_frames_tensor, rigid_mask)
    #  这个是根据我利用重构的frame算出来的，
    #  要和从 pdb里面直接拿到的atom position做一个比较
    gt_14pos_test = frame_to_14pos(gt_frames_tensor[..., 1:, :, :],
                                   gt_frames_tensor[..., :1, :, :],
                                   seq,
                                   coords)

    k = 32 if len(node_feature) >= 32 else len(node_feature)
    distance_rbf, relative_pos, edge_index = update_edges(init_rigid, seq, rigid_mask, k)

    # 暂时用这个function去算 atom 14 mask 之后再改
    atom14_atom_exists, gt_14pos = make_atom14_positions(seq, all_atom_positions)

    data = torch_geometric.data.Data(x=node_feature,
                                     esm_s=esm_s,
                                     aatype=seq,
                                     bb_coord=coords,
                                     rigid_mask=rigid_mask,
                                     fname=fname,
                                     edge_index=edge_index,
                                     edge_attr=relative_pos,
                                     atom_mask=atom_mask,
                                     bb_mask=bb_mask,
                                     gt_rigids=gt_frames_tensor,
                                     rigid=init_rigid,
                                     gt_14pos=gt_14pos,
                                     atom14_atom_exists=atom14_atom_exists,
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
            if len(protein['seq']) < 2000:
                proteins.append(protein_to_graph(protein))

        if graph_data:
            print("Store_data at", graph_data)
            with open(graph_data, "wb") as f:
                pickle.dump(proteins, f)
                print('finish data preprocess')

    return proteins


class ProteinDataset(Dataset, ABC):
    def __init__(self, data):
        super(ProteinDataset, self).__init__()
        self.proteins = data

    def len(self): return len(self.proteins)

    def get(self, item):
        protein = self.proteins[item]
        return copy.deepcopy(protein)
