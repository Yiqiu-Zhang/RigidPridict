from typing import Dict
import torch
from torch import Tensor

from rigid_predict.utlis import geometry
from rigid_predict.utlis.geometry import Rigid


def fape_loss(
        out: Dict[str, torch.Tensor],
        data,
        backbone_weight=0.5,
):
    loss = 0
    for batch in range(data.batch_size):

        mask = data.x_batch == batch
        gt_rigid_batch = data.gt_rigids[mask]
        pred_frames_batch = out['frames'][:, mask]
        atom_rigid_mask_batch = data.atom_mask[mask]

        bb_loss = backbone_loss(
            gt_rigid_batch,
            atom_rigid_mask_batch,
            traj=pred_frames_batch,
        )
        '''
        sc_loss = sidechain_loss(
            out["sidechain_frames"],
            out["positions"],
        )
        '''

        loss = loss + backbone_weight * bb_loss  # + config.sidechain.weight * sc_loss

    # Average over the batch dimension
    loss = loss / data.batch_size

    return loss

def backbone_loss(
        gt_rigid_tensor: torch.Tensor,
        atom_rigid_mask: torch.Tensor,
        traj: geometry.Rigid,

        use_clamped_fape=None,
        clamp_distance: float = 10.0,
        loss_unit_distance: float = 10.0,
        eps: float = 1e-4,
        **kwargs,
) -> torch.Tensor:
    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
    # backbone tensor, normalizes it, and then turns it back to a rotation
    # matrix. To avoid a potentially numerically unstable rotation matrix
    # to quaternion conversion, we just use the original rotation matrix
    # outright. This one hasn't been composed a bunch of times, though, so
    # it might be fine.
    gt_aff = geometry.from_tensor_4x4(gt_rigid_tensor)
    pred_aff = geometry.from_tensor_4x4(traj)
    fape_loss = compute_fape(
        pred_aff,
        gt_aff[None],  # add the cycle dimension
        atom_rigid_mask[None],
        pred_aff.trans,
        gt_aff[None].trans,
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )

    if use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            atom_rigid_mask[None],
            pred_aff.trans,
            gt_aff[None].trans,
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
        )

        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
                1 - use_clamped_fape
        )

    return fape_loss


def compute_fape(
        pred_frames: Rigid,
        target_frames: Rigid,
        frames_mask: torch.Tensor,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        length_scale: float,
        l1_clamp_distance: float = None,
        eps=1e-8,
) -> Tensor:
    """
        Computes FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # [*, N_frames, N_pts, 3] [N_cycle, N_rigid,] # [1, N_rigid, 3]
    local_pred_pos = geometry.rigid_mul_vec(pred_frames.invert(), pred_positions)

    # [1, N_rigid]# [1, N_rigid, 3]
    local_target_pos = geometry.rigid_mul_vec(target_frames.invert(), target_positions)

    # [N_cycle, N_rigid]
    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)
    # [N_cycle]
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))

    normed_error = torch.sum(normed_error, dim=-1)

    return normed_error
