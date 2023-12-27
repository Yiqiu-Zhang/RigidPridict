from typing import Dict
import torch
from rigid_predict.utlis import geometry
from rigid_predict.utlis.geometry import Rigid

def fape_loss(
        out: Dict[str, torch.Tensor],
        data,
        backbone_weight=0.5,
) -> torch.Tensor:

    bb_loss = backbone_loss(
        data.gt_rigids,
        traj=out["frames"],
    )
    '''
    sc_loss = sidechain_loss(
        out["sidechain_frames"],
        out["positions"],
    )
    '''

    loss = backbone_weight * bb_loss #+ config.sidechain.weight * sc_loss
    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss

def backbone_loss(
    gt_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: geometry.Rigid,
    use_clamped_fape = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    follow_batch =1,
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
    loss = []
    for i in follow_batch:
        fape_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            backbone_rigid_mask[None],
            pred_aff.trans,
            gt_aff[None].trans,
            backbone_rigid_mask[None],
            l1_clamp_distance=clamp_distance,
            length_scale=loss_unit_distance,
            eps=eps,
        )

        if use_clamped_fape is not None:
            unclamped_fape_loss = compute_fape(
                pred_aff,
                gt_aff[None],
                backbone_rigid_mask[None],
                pred_aff.get_trans(),
                gt_aff[None].get_trans(),
                backbone_rigid_mask[None],
                l1_clamp_distance=None,
                length_scale=loss_unit_distance,
                eps=eps,
            )

            fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
                1 - use_clamped_fape
            )

        loss.append(fape_loss)
    # Average over the batch dimension

    loss = torch.mean(loss)

    return loss

def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: float = None,
    eps=1e-8,
) -> torch.Tensor:
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

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = (
        normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    )
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error