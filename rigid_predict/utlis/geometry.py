from __future__ import annotations

import torch
import torch.nn.functional as F
from functools import lru_cache
from typing import Tuple, Any, Sequence, Optional

import numpy as np

_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}

def _to_mat(pairs):
    mat = np.zeros((4, 4))
    for pair in pairs:
        key, value = pair
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat

_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])


def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
        Converts a quaternion to a rotation matrix.

        Args:
            quat: [*, 4] quaternions
        Returns:
            [*, 3, 3] rotation matrices
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3]
    mat = _get_quat("_QTR_MAT", dtype=quat.dtype, device=quat.device)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))


def rot_to_quat(
    rot: torch.Tensor,
):
    if(rot.shape[-2:] != (3, 3)):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    k = [
        [ xx + yy + zz,      zy - yz,      xz - zx,      yx - xy,],
        [      zy - yz, xx - yy - zz,      xy + yx,      xz + zx,],
        [      xz - zx,      xy + yx, yy - xx - zz,      yz + zy,],
        [      yx - xy,      xz + zx,      yz + zy, zz - xx - yy,]
    ]

    k = (1./3.) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)

    _, vectors = torch.linalg.eigh(k)
    return vectors[..., -1]


_QUAT_MULTIPLY = np.zeros((4, 4, 4))
_QUAT_MULTIPLY[:, :, 0] = [[ 1, 0, 0, 0],
                          [ 0,-1, 0, 0],
                          [ 0, 0,-1, 0],
                          [ 0, 0, 0,-1]]

_QUAT_MULTIPLY[:, :, 1] = [[ 0, 1, 0, 0],
                          [ 1, 0, 0, 0],
                          [ 0, 0, 0, 1],
                          [ 0, 0,-1, 0]]

_QUAT_MULTIPLY[:, :, 2] = [[ 0, 0, 1, 0],
                          [ 0, 0, 0,-1],
                          [ 1, 0, 0, 0],
                          [ 0, 1, 0, 0]]

_QUAT_MULTIPLY[:, :, 3] = [[ 0, 0, 0, 1],
                          [ 0, 0, 1, 0],
                          [ 0,-1, 0, 0],
                          [ 1, 0, 0, 0]]

_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]

_CACHED_QUATS = {
    "_QTR_MAT": _QTR_MAT,
    "_QUAT_MULTIPLY": _QUAT_MULTIPLY,
    "_QUAT_MULTIPLY_BY_VEC": _QUAT_MULTIPLY_BY_VEC
}

@lru_cache(maxsize=None)
def _get_quat(quat_key, dtype, device):
    return torch.tensor(_CACHED_QUATS[quat_key], dtype=dtype, device=device)

def quat_multiply_by_vec(quat, vec):
    """Multiply a quaternion by a pure-vector quaternion."""
    mat = _get_quat("_QUAT_MULTIPLY_BY_VEC", dtype=quat.dtype, device=quat.device)
    reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
    return torch.sum(
        reshaped_mat *
        quat[..., :, None, None] *
        vec[..., None, :, None],
        dim=(-3, -2)
    )

def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
        Converts a quaternion to a rotation matrix.

        Args:
            quat: [*, 4] quaternions
        Returns:
            [*, 3, 3] rotation matrices
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3]
    mat = _get_quat("_QTR_MAT", dtype=quat.dtype, device=quat.device)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))

@lru_cache(maxsize=None)
def identity_rot_mats(
    shape: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    rots = torch.eye(
        3, dtype=dtype, device=device, requires_grad=requires_grad
    )
    rots = rots.view(*((1,) * len(shape)), 3, 3)
    rots = rots.expand(*shape, -1, -1)
    rots = rots.contiguous()

    return rots


@lru_cache(maxsize=None)
def identity_trans(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    trans = torch.zeros(
        (*batch_dims, 3),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )
    return trans


@lru_cache(maxsize=None)
def identity_quats(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    quat = torch.zeros(
        (*batch_dims, 4),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )

    with torch.no_grad():
        quat[..., 0] = 1

    return quat


class Rotation:

    def __init__(self,
                 rot_mats: Optional[torch.Tensor] = None,
                 quats: Optional[torch.Tensor] = None,
                 normalize_quats: bool = True,
                 ):

        # Force full-precision
        if quats is not None:
            quats = quats.to(dtype=torch.float32)
        if rot_mats is not None:
            rot_mats = rot_mats.to(dtype=torch.float32)

        if quats is not None and normalize_quats:
            quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)

        self._rot_mats = rot_mats
        self._quats = quats

    # Magic methods
    def __getitem__(self, index):

        if type(index) != tuple:
            index = (index,)

        if self._rot_mats is not None:
            rot_mats = self._rot_mats[index + (slice(None), slice(None))]
            return Rotation(rot_mats=rot_mats)
        elif self._quats is not None:
            quats = self._quats[index + (slice(None),)]
            return Rotation(quats=quats, normalize_quats=False)
        else:
            raise ValueError("rotation are None")
        
    def __mul__(self,
        right: torch.Tensor,
    ) -> Rotation:
        """
            Pointwise left multiplication of the transformation with a tensor.
            Can be used to e.g. mask the Rotation.

            Args:
                right:
                    The tensor multiplicand
            Returns:
                The product
        """
        if not(isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")
        
        if self._rot_mats is not None:

            new_rots = self._rot_mats * right[..., None, None] # [128,1,8,3,3] * [128, 14, 8] (3,3)
            return Rotation(rot_mats=new_rots)
        elif self._quats is not None:
            quats = self._quats * right[..., None] # all zero quats means nothing?
            return Rotation(quats=quats, normalize_quats=False)

    def get_rot_mat(self) -> torch.Tensor:
        """
        Return the underlying tensor rather than the Rotation object
        """
        rot_mats = self._rot_mats
        if rot_mats is None:
            if self._quats is None:
                raise ValueError("Both rotations are None")
            else:
                rot_mats = quat_to_rot(self._quats)

        return rot_mats

    def transpose(self):

        return torch.transpose(self._rot_mats, -2, -1)

    def invert(self) -> Rotation:
        """
            Returns the inverse of the current Rotation.

            Returns:
                The inverse of the current Rotation
        """

        if self._rot_mats is not None:
            return Rotation(
                rot_mats=invert_rot_mat(self._rot_mats),
                quats=None
            )
        else:
            raise ValueError("Both rotations are None")

    def unsqueeze(self, dim: int) -> Rotation:

        if dim >= len(self.shape):
            raise ValueError("Invalid dimension for Rotation object")

        if self._rot_mats is not None:
            rot_mats = self._rot_mats.unsqueeze(dim if dim >= 0 else dim - 2)
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = self._quats.unsqueeze(dim if dim >= 0 else dim - 1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    @property
    def shape(self) -> torch.Size:
        """
        Returns the virtual shape of the rotation object.
        If the Rotation was initialized with a [10, 3, 3]
        rotation matrix tensor, for example, the resulting shape would be
        [10].
        """
        s = self._rot_mats.shape[:-2]

        return s

    @staticmethod
    def identity(shape,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 requires_grad: bool = True,
                 fmt: str = "rot_mat",
                 ) -> Rotation:

        if(fmt == "rot_mat"):
            rot_mats = identity_rot_mats(
                shape, dtype, device, requires_grad,
            )
            return Rotation(rot_mats=rot_mats, quats=None)
        elif(fmt == "quat"):
            quats = identity_quats(shape, dtype, device, requires_grad)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError(f"Invalid format: f{fmt}")

        return Rotation(rot_mats)

    def get_quats(self) -> torch.Tensor:
        """
            Returns the underlying rotation as a quaternion tensor.

            Depending on whether the Rotation was initialized with a
            quaternion, this function may call torch.linalg.eigh.

            Returns:
                The rotation as a quaternion tensor.
        """
        if self.get_rot_mat() is None:
            raise ValueError("Both rotations are None")
        else:
            quats = rot_to_quat(self.get_rot_mat())

        return quats

    def compose_q_update_vec(self,
        q_update_vec: torch.Tensor,
        normalize_quats: bool = True
    ) -> Rotation:
        """
            Returns a new quaternion Rotation after updating the current
            object's underlying rotation with a quaternion update, formatted
            as a [*, 3] tensor whose final three columns represent x, y, z such
            that (1, x, y, z) is the desired (not necessarily unit) quaternion
            update.

            Args:
                q_update_vec:
                    A [*, 3] quaternion update tensor
                normalize_quats:
                    Whether to normalize the output quaternion
            Returns:
                An updated Rotation
        """
        quats = self.get_quats()
        new_quats = quats + quat_multiply_by_vec(quats, q_update_vec)

        return Rotation(
            rot_mats=None,
            quats=new_quats,
            normalize_quats=normalize_quats,
        )

class Rigid:
    """
        A class representing a rigid transformation. Little more than a wrapper
        around two objects: a Rotation object and a [*, 3] translation
        Designed to behave approximately like a single torch tensor with the
        shape of the shared batch dimensions of its component parts.
    """

    def __init__(self,
                 rots: Rotation,
                 trans: Optional[torch.Tensor],
                 loc: torch.Tensor
                 ):
        if trans is None:
            batch_dims = rots.shape
            dtype = rots.get_rot_mat().dtype
            device = rots.get_rot_mat().device
            requires_grad = rots.get_rot_mat().requires_grad
            trans = identity_trans(batch_dims, dtype, device, requires_grad)
            loc = identity_trans(batch_dims, dtype, device, requires_grad)


        self.rot = rots
        self.trans = trans
        self.loc = loc

    @staticmethod # 不要求class被实例化，可以像 xx.py xx.identity 一样调用
    def identity(shape: Tuple[int],
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 requires_grad: bool = True, # 应该是False吧，这里好像不用加梯度
                 ) -> Rigid:
        """
            Constructs an identity transformation.

            Args:
                shape:
                    The desired shape
                dtype:
                    The dtype of both internal tensors
                device:
                    The device of both internal tensors
                requires_grad:
                    Whether grad should be enabled for the internal tensors
            Returns:
                The identity transformation
        """

        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad),
            identity_trans(shape, dtype, device, requires_grad),
            identity_trans(shape, dtype, device, requires_grad),
        )

    def __getitem__(self, index: Any) -> Rigid:

        if type(index) != tuple:
            index = (index,)

        return Rigid(self.rot[index], self.trans[index + (slice(None),)], self.loc[index + (slice(None),)])
    
    def __mul__(self,
        right: torch.Tensor,
    ) -> Rigid:
        """
            Pointwise left multiplication of the transformation with a tensor.
            Can be used to e.g. mask the Rigid.

            Args:
                right:
                    The tensor multiplicand
            Returns:
                The product
        """
        if not(isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        new_rots = self.rot * right
        new_trans = self.trans * right[..., None]
        new_loc = self.loc * right[...,None]
        return Rigid(new_rots, new_trans,new_loc)

    def edge(self, edge_index):

        """
        Forming fully connected graph edge using the rigid object
        """
        rot_T = self.rot[edge_index[0]].transpose()
        rot = self.rot[edge_index[1]].get_rot_mat()
        orientation = torch.einsum('mij,mjk->mik', rot_T, rot)

        displacement =  self.loc[edge_index[0]] - self.loc[edge_index[1]]
        distance = torch.linalg.vector_norm(displacement,dim=-1).float() 
        direction = F.normalize(displacement,dim=-1).float()
        altered_direction = rot_vec(rot_T, direction) # why write this? dont know why

        return distance, altered_direction, orientation

    def unsqueeze(self, dim: int) -> Rigid:

        if dim >= len(self.shape):
            raise  ValueError("Invalid dimension for Rigid object")

        rot = self.rot.unsqueeze(dim)
        trans = self.trans.unsqueeze(dim if dim >=0 else dim -1)
        loc = self.loc.unsqueeze(dim if dim >=0 else dim -1)
        return Rigid(rot, trans, loc)

    @property
    def shape(self) -> torch.Size:

        s = self.trans.shape[:-1]
        return s
    
    @property
    def device(self) -> torch.device:
        """
            The device of the underlying rotation

            Returns:
                The device of the underlying rotation
        """
        if self.rot.get_rot_mat() is not None:
            assert self.rot.get_rot_mat().device == self.loc.device
            return self.loc.device
        else:
            raise ValueError("Both rotations are None")

    def cuda(self) -> Rigid:
        """
            Moves the transformation object to GPU memory
            
            Returns:
                A version of the transformation on GPU
        """
        return Rigid(Rotation(self.rot.get_rot_mat().cuda()), self.trans.cuda(),self.loc.cuda())

    def stop_rot_gradient(self) -> Rigid:
        """
            Detaches the underlying rotation object

            Returns:
                A transformation object with detached rotations
        """
        fn = lambda r: r.detach()
        return self.apply_rot_fn(fn)

    def apply_rot_fn(self, fn) -> Rigid:
        """
            Applies a Rotation -> Rotation function to the stored rotation
            object.

            Args:
                fn: A function of type Rotation -> Rotation
            Returns:
                A transformation object with a transformed rotation.
        """
        return Rigid(fn(self.rot), self.trans)

    @staticmethod
    def from_3_points(
        p_neg_x_axis: torch.Tensor,
        origin: torch.Tensor,
        p_xy_plane: torch.Tensor,
        eps: float = 1e-8
    ) -> Rigid:
        """
            Implements algorithm 21. Constructs transformations from sets of 3
            points using the Gram-Schmidt algorithm.
            Only used for the sampling process, so ignore Rigid loc project, giving it random values.
            Args:
                p_neg_x_axis: [*, 3] coordinates
                origin: [*, 3] coordinates used as frame origins
                p_xy_plane: [*, 3] coordinates
                eps: Small epsilon value
            Returns:
                A transformation object of shape [*]
        """
        p_neg_x_axis = torch.unbind(p_neg_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        rot_obj = Rotation(rot_mats=rots)

        return Rigid(rot_obj, torch.stack(origin, dim=-1), torch.stack(origin, dim=-1))

    def invert(self) -> Rigid:
        """
            Inverts the transformation.

            Returns:
                The inverse transformation.
        """
        rot_inv = self.rot.invert()
        trn_inv = rot_vec(rot_inv, self.trans)

        return Rigid(rot_inv, -1 * trn_inv)

    def compose_q_update_vec(self,
        q_update_vec: torch.Tensor,
    ) -> Rigid:
        """
            Composes the transformation with a quaternion update vector of
            shape [*, 6], where the final 6 columns represent the x, y, and
            z values of a quaternion of form (1, x, y, z) followed by a 3D
            translation.

            Args:
                q_vec: The quaternion update vector.
            Returns:
                The composed transformation.
        """
        q_vec, t_vec = q_update_vec[..., :3], q_update_vec[..., 3:]
        new_rots = self.rot.compose_q_update_vec(q_vec)

        trans_update = rot_vec(self.rot.get_rot_mat(), t_vec)
        new_translation = self.trans + trans_update

        return Rigid(new_rots, new_translation)

def from_tensor_4x4(t: torch.Tensor) -> Rigid:
    """
        Constructs a transformation from a homogenous transformation
        tensor.

        Args:
            t: [*, 4, 4] homogenous transformation tensor
        Returns:
            T object with shape [*]
    """
    if t.shape[-2:] != (5, 5):
        raise ValueError("Incorrectly shaped input tensor")

    rots = Rotation(rot_mats=t[..., :3, :3],)
    trans = t[..., :3, 3]
    loc = t[...,:3, 4]
    return Rigid(rots, trans, loc)

def to_tensor_5x5(r: Rigid) ->  torch.Tensor:
    r_tensor = torch.zeros(*r.trans.shape[:-1],5,5)
    r_tensor[...,:3,:3] = r.rot.get_rot_mat()
    r_tensor[...,:3, 3] = r.trans
    r_tensor[...,:3, 4] = r.loc
    return r_tensor

@lru_cache(maxsize=None)
def identity_trans(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    trans = torch.zeros(
        (*batch_dims, 3),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )
    return trans

def Rigid_mult(rigid_1: Rigid,
               rigid_2: Rigid) -> Rigid:
    rot1 = rigid_1.rot.get_rot_mat()
    rot2 = rigid_2.rot.get_rot_mat().to(rot1.device)

    new_rot = rot_matmul(rot1, rot2)
    new_trans = rot_vec(rot1, rigid_2.trans.to(rot1.device))  + rigid_1.trans.to(rot1.device)
    new_loc = rot_vec(rot1, rigid_2.loc.to(rot1.device)) + rigid_1.trans.to(rot1.device)

    return  Rigid(Rotation(new_rot), new_trans, new_loc)

def Rigid_update_trans(rigid: Rigid,
                       t_vec: torch.Tensor):

    updated_loc = loc_rigid_mul_vec(rigid,t_vec)

    return Rigid(rigid.rot, rigid.trans, updated_loc)
    #updated_trans = rigid_mul_vec(rigid,t_vec)

    #return Rigid(rigid.rot, updated_trans)

def flatten_rigid(rigid: Rigid) -> Rigid:
    flat_rot = rigid.rot.get_rot_mat().flatten(start_dim=-4, end_dim=-3)
    flat_trans = rigid.trans.flatten(start_dim=-3, end_dim=-2)
    flat_loc = rigid.loc.flatten(start_dim=-3, end_dim=-2)
    return Rigid(Rotation(flat_rot), flat_trans, flat_loc)

def unflatten_toN4(r: Rigid) -> Rigid:

    rot = r.rot.get_rot_mat().reshape(-1,5,3,3)[...,1:]
    trans = r.trans.reshape(-1,5,3,3)[...,1:]
    loc = r.loc.reshape(-1,5,3,3)[...,1:]

    return Rigid(Rotation(rot_mats=rot), trans, loc)

def rigid_mul_vec(rigid: Rigid,
                  vec: torch.Tensor) -> torch.Tensor:

    rot_mat = rigid.rot.get_rot_mat()

    rotated = rot_vec(rot_mat, vec)

    return rotated + rigid.trans

def loc_rigid_mul_vec(rigid: Rigid,
                      vec: torch.Tensor) -> torch.Tensor:
    '''
    Different from rigid_mul_vec, This one is used for attention and distance calculation,
    which is NOT related to the position calculation. To calculate the position of the
    final atom, use rigid_mul_vec
    '''
    rot_mat = rigid.rot.get_rot_mat()
    rotated = rot_vec(rot_mat, vec)
    return rotated + rigid.loc

def invert_rot_mat(rot_mat: torch.Tensor):
    return rot_mat.transpose(-1, -2)
def loc_invert_rot_mul_vec(rigid: Rigid,
                         vec: torch.Tensor) -> torch.Tensor:
    """
        The inverse of the apply() method.

        Args:
            pts:
                A [*, 3] set of points
        Returns:
            [*, 3] inverse-rotated points
    """
    rot_mats = rigid.rot.get_rot_mat()
    inv_rot_mats = rot_mats.transpose(-1, -2)
    final_vec = rot_vec(inv_rot_mats, vec - rigid.loc)

    return final_vec

def invert_rot_mul_vec(rigid: Rigid,
                         vec: torch.Tensor) -> torch.Tensor:
    """
        The inverse of the apply() method.

        Args:
            pts:
                A [*, 3] set of points
        Returns:
            [*, 3] inverse-rotated points
    """
    rot_mats = rigid.rot.get_rot_mat()
    inv_rot_mats = rot_mats.transpose(-1, -2)
    final_vec = rot_vec(inv_rot_mats, vec - rigid.trans)

    #inv_trans = rot_vec(inv_rot_mats, rigid.trans)
    #rotated = rot_vec(inv_rot_mats, vec)

    return final_vec # rotated - inv_trans

def rot_matmul(
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """
        Performs matrix multiplication of two rotation matrix tensors. Written
        out by hand to avoid AMP downcasting.

        Args:
            a: [*, 3, 3] left multiplicand
            b: [*, 3, 3] right multiplicand
        Returns:
            The product ab
    """
    def row_mul(i):
        return torch.stack(
            [
                a[..., i, 0] * b[..., 0, 0]
                + a[..., i, 1] * b[..., 1, 0]
                + a[..., i, 2] * b[..., 2, 0],
                a[..., i, 0] * b[..., 0, 1]
                + a[..., i, 1] * b[..., 1, 1]
                + a[..., i, 2] * b[..., 2, 1],
                a[..., i, 0] * b[..., 0, 2]
                + a[..., i, 1] * b[..., 1, 2]
                + a[..., i, 2] * b[..., 2, 2],
            ],
            dim=-1,
        )

    return torch.stack(
        [
            row_mul(0),
            row_mul(1),
            row_mul(2),
        ],
        dim=-2
    )

def rot_vec(
    r: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
        Applies a rotation to a vector. Written out by hand to avoid transfer
        to avoid AMP downcasting.

        Args:
            r: [*, 3, 3] rotation matrices
            t: [*, 3] coordinate tensors
        Returns:
            [*, 3] rotated coordinates
    """
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )

def cat(rigids: Sequence[Rigid],
        dim: int) -> Rigid:
    rot_mats = [r.rot.get_rot_mat() for r in rigids]
    rot_mats = torch.cat(rot_mats, dim=dim if dim >= 0 else dim - 2)
    rots = Rotation(rot_mats= rot_mats)

    trans = torch.cat([r.trans for r in rigids], dim=dim if dim >= 0 else dim - 1)
    loc = torch.cat([r.loc for r in rigids], dim=dim if dim >= 0 else dim - 1)
    return Rigid(rots, trans, loc)

def map_rigid_fn(rigid: Rigid):

    rot_mat = rigid.rot.get_rot_mat()
    rot_mat = rot_mat.view(rot_mat.shape[:-2] + (9,))
    rot_mat = torch.stack(list(map(
        lambda x: torch.sum(x, dim=-1), torch.unbind(rot_mat, dim=-1)
    )), dim=-1)
    rot_mat = rot_mat.view(rot_mat.shape[:-1] + (3,3))

    new_trans = torch.stack(list(map(
        lambda x: torch.sum(x, dim=-1), torch.unbind(rigid.trans, dim=-1)
    )), dim=-1)
    fake_loc = torch.zeros(new_trans.shape)
    return Rigid(Rotation(rot_mats=rot_mat), new_trans, fake_loc)

def get_gb_trans(bb_pos: torch.Tensor) -> Rigid: # [*,128,4,3]

    '''
    Get global transformation from given backbone position
    '''
    ex = bb_pos[..., 2, :] - bb_pos[..., 1, :] # [*,128,3] C - CA
    y_vec = bb_pos[..., 0, :] - bb_pos[..., 1, :] # [*,128,3] N - CA
    t = bb_pos[..., 1, :] # [*,128,3] CA
    
   # print("ex====",ex.shape)
   # print("y_vec====",y_vec.shape)
   # print("t====",t.shape)
   # print("torch.linalg.vector_norm(ex, dim=-1)", torch.linalg.vector_norm(ex, dim=-1).shape)
    # [*, N, 3]
    ex_norm = ex / torch.linalg.vector_norm(ex, dim=-1).unsqueeze(-1)
  #  print(ex_norm.shape)
    def dot(a, b):  # [*, N, 3]
        x1, y1, z1 = torch.unbind(b, dim=-1)
        x2, y2, z2 = torch.unbind(a, dim=-1)

        return x1 * x2 + y1 * y2 + z1 * z2

    ey = y_vec - dot(y_vec, ex_norm).unsqueeze(-1) * ex_norm
    ey_norm = ey / torch.linalg.norm(ey, dim=-1).unsqueeze(-1)

    ez_norm = torch.cross(ex_norm, ey_norm, dim=-1)

    '''
    m = torch.stack([ex_norm, ey_norm, ez_norm, t], dim=-1)
    m = torch.transpose(m, -2, -1)
    last_dim = torch.tensor([[0.0], [0.0], [0.0], [1.0]]).expand(6,12,4,1)

    m = torch.cat([m, last_dim], axis=-1)
    '''

    new_rot = torch.nan_to_num(torch.stack([ex_norm, ey_norm, ez_norm], dim=-1))
    return Rigid(Rotation(rot_mats=new_rot),
                 t,
                 torch.zeros(t.shape, device=t.device)
                 )