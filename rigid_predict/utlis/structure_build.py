import torch
import torch.nn as nn
import numpy as np
from rigid_predict.utlis.constant import (restype_rigid_group_default_frame,
                                      restype_atom14_to_rigid_group,
                                      restype_atom14_mask,
                                      restype_atom14_rigid_group_positions,
                                      restype_atom37_mask,
                                      make_atom14_37_list,
                                      )
from rigid_predict.utlis import constant as constant
import rigid_predict.utlis.constant as rc
import rigid_predict.utlis.geometry as geometry
import rigid_predict.utlis.protein as protein
import os

device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def rotate_sidechain(
        restype_idx: torch.Tensor,  # [N]
        angles: torch.Tensor,  # [N,4，2]
        last_local_r: geometry.Rigid,  # [N, 8] Rigid
) -> geometry.Rigid:
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)

    # [N,4] + [N,4] == [N,8]
    # adding 4 zero angles which means no change to the default value.
    sin_angles = torch.cat([torch.zeros(*restype_idx.shape, 4).to(sin_angles.device), sin_angles], dim=-1)
    cos_angles = torch.cat([torch.ones(*restype_idx.shape, 4).to(sin_angles.device), cos_angles], dim=-1)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.
    all_rots = sin_angles.new_zeros(last_local_r.rot.get_rot_mat().shape).to(sin_angles.device)
    # print("orign all_rots==",all_rots.shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = cos_angles
    all_rots[..., 1, 2] = -sin_angles
    all_rots[..., 2, 1] = sin_angles
    all_rots[..., 2, 2] = cos_angles

    all_rots = geometry.Rigid(geometry.Rotation(rot_mats=all_rots), None)

    all_frames = geometry.Rigid_mult(last_local_r, all_rots)

    # Rigid
    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = geometry.Rigid_mult(chi1_frame_to_bb, chi2_frame_to_frame)
    chi3_frame_to_bb = geometry.Rigid_mult(chi2_frame_to_bb, chi3_frame_to_frame)
    chi4_frame_to_bb = geometry.Rigid_mult(chi3_frame_to_bb, chi4_frame_to_frame)

    all_frames_to_bb = geometry.cat(
        [all_frames[..., :5],
         chi2_frame_to_bb.unsqueeze(-1),
         chi3_frame_to_bb.unsqueeze(-1),
         chi4_frame_to_bb.unsqueeze(-1), ],
        dim=-1,
    )

    return all_frames_to_bb, all_frames


def frame_to_14pos(frames, gt_frame, aatype_idx, bb_cords):

    frames = torch.cat([gt_frame, frames], dim=-3)
    frames = geometry.from_tensor_4x4(frames)
    # [21 , 14]
    group_index = torch.tensor(restype_atom14_to_rigid_group).to(frames.device).to(torch.int64)

    # [21 , 14] idx [*, N] -> [*, N, 14]
    group_mask = group_index[aatype_idx, ...]
    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(group_mask, num_classes=frames.shape[-1])

    # [*, N, 14, 8] Rigid frames for every 14 atoms, non exist atom are mapped to group 0
    map_atoms_to_global = frames[..., None, :] * group_mask  # [*, N, :, 8] * [*, N, 14, 8]

    # [*, N, 14]
    map_atoms_to_global = geometry.map_rigid_fn(map_atoms_to_global)

    # [21 , 14]
    atom_mask = torch.tensor(restype_atom14_mask).to(frames.device)
    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype_idx, ...].unsqueeze(-1)

    # [21, 14, 3] # 这个是以sidechain frame 为标准设立的pos， 按我的逻辑来肯定不能从这里开始移动
    default_pos = torch.tensor(restype_atom14_rigid_group_positions).to(frames.device)
    # [*, N, 14, 3]
    default_pos = default_pos[aatype_idx, ...]

    pred_pos = geometry.rigid_mul_vec(map_atoms_to_global, default_pos)
    pred_pos = pred_pos * atom_mask

    #pred_pos, _ = atom14_to_atom37_batched(pred_pos, aatype_idx)

    # this is just for convinient use of the backbone coordinate
    # [B, N, 37, 3] [B,N,4,3]
    pred_pos[..., :4, :] = bb_cords[..., :4, :]

    return pred_pos

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)  # torch.arange N
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))  # [N, 1]
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)  # [Tensor(N,1), Tensor(N,37), slice(None)]
    return data[ranges]  # [N, 37, 3]

def atom14_to_atom37_batched(atom14, aa_idx):  # atom14: [*, N, 14, 3]

    restype_atom37_to_atom14 = make_atom14_37_list()  # 注意有错

    residx_atom37_to_14 = restype_atom37_to_atom14[aa_idx]

    # [N, 37]
    atom37_mask = torch.tensor(restype_atom37_mask).to(atom14.device)
    atom37_mask = atom37_mask[aa_idx]

    # [N, 37, 3]
    atom37 = batched_gather(atom14,
                            residx_atom37_to_14,
                            dim=-2,
                            no_batch_dims=len(atom14.shape[:-2])
                            )
    atom37 = atom37 * atom37_mask[..., None]

    return atom37, atom37_mask

def batch_gather(data,  # [N, 14, 3]
                 indexing):  # [N,37]
    ranges = []

    N = data.shape[-3]
    r = torch.arange(N)
    # print("r1========",r.shape)
    r = r.view(-1, 1)
    # print("r2========",r.shape)
    ranges.append(r)
    # print("r3========",r.shape)
    remaining_dims = [slice(None) for _ in range(2)]
    # print("remaining_dims1========",remaining_dims.shape)
    remaining_dims[-2] = indexing
    # print("remaining_dims2========",remaining_dims.shape)
    ranges.extend(remaining_dims)  # [Tensor(N,1), Tensor(N,37), slice(None)]
    # print("ranges========",ranges.shape)
    return data[ranges]  # [N, 37, 3]

def atom14_to_atom37(atom14, aa_idx):  # atom14: [*, N, 14, 3]

    restype_atom37_to_atom14 = make_atom14_37_list()  # 注意有错

    residx_atom37_to_14 = restype_atom37_to_atom14[aa_idx]
    # [N, 37]
    atom37_mask = torch.tensor(restype_atom37_mask)
    atom37_mask = atom37_mask[aa_idx]

    # [N, 37, 3]
    # print('atom14===========', atom14.shape)
    # print('atom37_mask===========', atom37_mask.shape)
    # print('residx_atom37_to_14=====', residx_atom37_to_14.shape)
    atom37 = batch_gather(atom14, residx_atom37_to_14)
    atom37 = atom37 * atom37_mask[..., None]

    return atom37

def get_default_r(restype_idx):
    default_frame = torch.tensor(restype_rigid_group_default_frame)

    # [*, N, 8, 4, 4]
    res_default_frame = default_frame[restype_idx, ...]

    # [*, N, 8] Rigid
    default_r = geometry.from_tensor_4x4(res_default_frame)
    return default_r


def torsion_to_frame(angles,
                     protein
                     ):  # -> [*, N, 5] Rigid
    """Compute all residue frames given torsion
        angles and the fixed backbone coordinates.

        Args:
            aatype_idx: aatype for each residue
            backbone_position: backbone coordinate for each residue
            angles: torsion angles for each residue

        return:
            all frames [N, 5] Rigid
        """

    bb_to_gb = geometry.get_gb_trans(protein.bb_coord)

    sc_to_bb, local_r = rotate_sidechain(protein.aatype, angles, protein.local_rigid)
    all_frames_to_global = geometry.Rigid_mult(bb_to_gb[..., None], sc_to_bb)

    # [N_rigid] Rigid
    flatten_frame = geometry.flatten_rigid(all_frames_to_global[..., [0, 4, 5, 6, 7]])

    flat_rigids = flatten_frame[protein.rigid_mask]

    return flat_rigids, local_r, all_frames_to_global

def get_gt_init_frames(angles, bb_coord, aatype, rigid_mask):

    init_local_rigid = get_default_r(aatype)

    bb_to_gb = geometry.get_gb_trans(bb_coord)
    sc_to_bb, local_r = rotate_sidechain(aatype, angles, init_local_rigid)
    gt_global_frame = geometry.Rigid_mult(bb_to_gb[..., None], sc_to_bb)

    init_frame  = geometry.Rigid_mult(bb_to_gb[..., None], geometry.Rigid.identity(sc_to_bb.shape, requires_grad=False))

    # [N_rigid] Rigid
    flatten_frame = geometry.flatten_rigid(gt_global_frame[..., [0, 4, 5, 6, 7]])
    init_rigid = geometry.flatten_rigid(init_frame[..., [0, 4, 5, 6, 7]])

    flat_rigids = flatten_frame[rigid_mask]
    init_rigid = init_rigid[rigid_mask]

    return flat_rigids.to_tensor_4x4(), local_r.to_tensor_4x4(), gt_global_frame.to_tensor_4x4(), init_rigid.to_tensor_4x4()


def update_E_idx(frames: geometry.Rigid,  # [*, N_rigid] Rigid
                 pair_mask: torch.Tensor,  # [*, N_res]
                 top_k: int,
                 ):
    # [*, N_rigid, N_rigid]
    distance, _, _ = frames.edge()

    D = pair_mask * distance

    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - pair_mask) * D_max  # give masked position value D_max

    # Value of distance [*, N_rigid, K], Index of distance [*, N_rigid, K]
    _, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)

    return E_idx


def write_pdb_from_position(graph, final_atom_positions, out_path, fname, j):
    final_atom_mask = restype_atom37_mask[graph.aatype.cpu()]
    chain_len = len(graph.aatype)
    index = np.arange(1, chain_len + 1)

    resulted_protein = protein.Protein(
        aatype=graph.aatype.cpu(),  # [*,N]
        atom_positions=final_atom_positions,
        atom_mask=final_atom_mask,
        residue_index=index,  # 0,1,2,3,4 range_chain
        b_factors=np.zeros_like(final_atom_mask))

    pdb_str = protein.to_pdb(resulted_protein)

    with open(os.path.join(out_path, f"{fname}_generate_{j}.pdb"), 'w') as fp:
        fp.write(pdb_str)


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.restypes:
        residue_name = rc.restype_1to3[residue_name]
        residue_chi_angles = rc.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([rc.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append(
                [0, 0, 0, 0]
            )  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return chi_atom_indices

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)  # torch.arange N
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))  # [N, 1]
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)  # [Tensor(N,1), Tensor(N,37), slice(None)]
    return data[ranges]  # [N, 37, 3]


def make_atom14_positions(aatypes):

    restype_atom14_mask = []

    for rt in constant.restypes:

        atom_names = constant.restype_name_to_atom14_names[
            constant.restype_1to3[rt]]
        
        restype_atom14_mask.append(
            [(1.0 if name else 0.0) for name in atom_names]
        )

    # Add dummy mapping for restype 'UNK'.
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_mask =torch.tensor(restype_atom14_mask, dtype=torch.long, device=aatypes.device)
    residx_atom14_mask = restype_atom14_mask[aatypes]

    return residx_atom14_mask
