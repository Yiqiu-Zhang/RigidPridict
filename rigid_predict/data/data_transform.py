import numpy as np
import torch
from rigid_predict.utlis import constant
from rigid_predict.utlis import constant_test
from rigid_predict.utlis.geometry import Rigid, Rotation
from rigid_predict.utlis import geometry


def atom37_to_frames(aatype,
                     all_atom_positions,
                     eps=1e-8):

    restype_rigidgroup_base_atom_names = np.full([21, 3, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]  # bb frame
    # restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"] # bb O position

    for restype, restype_letter in enumerate(constant.restypes):
        resname = constant.restype_1to3[restype_letter]
        for rigid_idx in range(2):
            if constant_test.frame_mask[restype][rigid_idx+1]:
                names = constant_test.frame_atoms[resname][rigid_idx]
                restype_rigidgroup_base_atom_names[
                restype, rigid_idx + 1, :
                ] = names

    lookuptable = constant_test.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    # [21, 3, 3]
    restype_rigidgroup_base_atom37_idx = lookup(
        restype_rigidgroup_base_atom_names,
    )
    restype_rigidgroup_base_atom37_idx = aatype.new_tensor(
        restype_rigidgroup_base_atom37_idx,
    )
    # [N_res, 3, 3]
    residx_rigidgroup_base_atom37_idx = restype_rigidgroup_base_atom37_idx[aatype]

    # [N_res, 3_frame, 3_atom, 3_coord]
    base_atom_pos = batched_gather(
        all_atom_positions,  # [N_res, 38, 3]
        residx_rigidgroup_base_atom37_idx,
        dim=-2,
        no_batch_dims=len(all_atom_positions.shape[:-2]),
    )

    gt_frames = Rigid.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=eps,
    )

    rots = torch.eye(3, dtype=all_atom_positions.dtype, device=aatype.device)
    rots = torch.tile(rots, (1, 3, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(rot_mats=rots)

    gt_frames = geometry.Rigid_mult(gt_frames, Rigid(rots, None))
    gt_frames_tensor = gt_frames.to_tensor_4x4()

    return gt_frames_tensor


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]
