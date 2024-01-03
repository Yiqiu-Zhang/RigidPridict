from typing import *
import os
import shutil
import glob
from pathlib import Path
import json
import logging
import time 

import torch
import torch.nn as nn
import numpy as np
from rigid_predict.utlis import geometry, structure_build
from rigid_predict.utlis.tensor_utlis import dict_multimap
from rigid_predict.model.attention import GraphIPA

class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):

        out = nn.functional.layer_norm(
            x,
            self.c_in,
            self.weight,
            self.bias,
            self.eps,
        )

        return out

class TransitionLayer(nn.Module):
    """ We only get one transitionlayer in our model, so no module needed."""
    def __init__(self, c):
        super(TransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = nn.Linear(self.c, self.c)
        self.linear_2 = nn.Linear(self.c, self.c)
        self.linear_3 = nn.Linear(self.c, self.c)

        self.relu = nn.ReLU()

        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        s = s
        s = self.layer_norm(s)

        return s

class EdgeTransition(nn.Module):
    def __init__(
            self,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = nn.Linear(node_embed_size, bias_embed_size)
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(nn.Linear(hidden_size, hidden_size))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = nn.Linear(hidden_size, edge_embed_out)
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, data, node_emb, edge_emb):

        # [N_rigid, c_n/2]
        node_emb = self.initial_embed(node_emb)

        # [E, c_n]
        edge_bias = torch.cat([node_emb[data.edge_index[0]], node_emb[data.edge_index[1]],], axis=-1)

        # [E, c_n + c_z]
        edge_emb = torch.cat([edge_emb, edge_bias], axis=-1)

        # [E, c_z]
        edge_emb1 = self.trunk(edge_emb)
        edge_emb = self.final_layer(edge_emb1 + edge_emb)
        edge_emb = self.layer_norm(edge_emb)

        return edge_emb

class RigidUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(RigidUpdate, self).__init__()

        self.linear = nn.Linear(c_s, 6)

    def forward(self,
                s: torch.Tensor,
                bb_mask: torch.Tensor,
                atom_mask: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:

        # [N, 6]
        update = self.linear(s)
        am = torch.ones(len(atom_mask), 6).to(s.device)
        bm = torch.ones(len(bb_mask), 6).to(s.device)
        am[atom_mask] = torch.tensor([0., 0, 0., 1., 1., 1.]).to(s.device) # only mask rotation
        bm[bb_mask] = torch.tensor([0., 0., 0., 0., 0., 0.]).to(s.device) # mask both rotation and translation

        update = update * am * bm

        return update

class PairEmbedder(nn.Module):
    """
    Embeds "template_pair_feat" features.

    Implements Algorithm 2, line 9.
    """

    def __init__(
            self,
            pair_dim,
            c_z,

    ):
        """
        Args:
            c_in:

            c_out:
                Output channel dimension
        """
        super(PairEmbedder, self).__init__()
        '''
        self.pair_stack = PairStack(c_z,
                                    c_hidden_tri_att,
                                    c_hidden_tri_mul,
                                    no_blocks,
                                    no_heads,
                                    pair_transition_n)
        '''
        # Despite there being no relu nearby, the source uses that initializer
        self.linear_1 = nn.Linear(pair_dim, c_z)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(c_z, c_z)
        self.ln = nn.LayerNorm(c_z)

    def forward(
            self,
            pair_feature: torch.Tensor,
    ) -> torch.Tensor:
        pair_emb = self.linear_1(pair_feature)
        pair_emb = self.relu(pair_emb)
        pair_emb = self.linear_2(pair_emb)

        pair_emb = self.ln(pair_emb)

        #  pair_emb = self.pair_stack(pair_emb, pair_mask)

        return pair_emb

class InputEmbedder(nn.Module):
    def __init__(
            self,
            nf_dim: int,  # Node features dim
            c_s: int,  # Node_embedding dim

            # Pair Embedder parameter
            pair_dim: int,  # Pair features dim
            c_z: int,  # Pair embedding dim

    ):
        super(InputEmbedder, self).__init__()
        self.nf_dim = nf_dim
        self.c_z = c_z
        self.c_n = c_s

        self.relpos_embedding = torch.nn.Embedding(pair_dim, c_z)

        self.pair_embedder = PairEmbedder(c_z,
                                          c_z,)

        self.linear_tf_0 = nn.Linear(nf_dim, c_s)
        self.linear_tf_1 = nn.Linear(c_s, c_s)
        self.relu = nn.ReLU()
        self.linear_tf_2 = nn.Linear(c_s, c_s)
        self.ln = nn.LayerNorm(c_s)

        self.proj = nn.Linear(c_s, c_z * 2, bias=True)
        self.o_proj = nn.Linear(2 * c_z, c_z, bias=True)

    def forward(self, data):

        ems_s = data.esm_s.repeat(1, 5).reshape(-1, data.esm_s.shape[-1])
        ems_s = ems_s[data.rigid_mask]

        s = ems_s + self.linear_tf_0(data.x)
        z = self.relpos_embedding(data.edge_attr)

        # [N_rigid, c_n]
        s = self.linear_tf_1(s)
        s = self.relu(s)
        s = self.linear_tf_2(s)
        s = self.ln(s)

        s_z = self.proj(s)
        q, k = s_z.chunk(2, dim=-1)
        q = q[data.edge_index[0]]
        k = k[data.edge_index[1]]

        prod = q * k
        diff = q - k
        s_z = torch.cat([prod, diff], dim=-1)

        z = z + self.o_proj(s_z)
        z = z + self.pair_embedder(z)

        return s, z

class StructureUpdateModule(nn.Module):

    def __init__(self,
                 no_blocks,
                 c_n,
                 c_z,
                 c_hidden,
                 ipa_no_heads,
                 no_qk_points,
                 no_v_points,
                 ):

        super(StructureUpdateModule, self).__init__()

        self.blocks = nn.ModuleList()
        self.relu = nn.ReLU()
        for block_idx in range(no_blocks):
            block = StructureBlock(c_n,
                                   c_z,
                                   c_hidden,
                                   ipa_no_heads,
                                   no_qk_points,
                                   no_v_points,
                                   block_idx,
                                   no_blocks,
                                   )

            self.blocks.append(block)



    def forward(self, data, s, z):

        outputs = []
        r = geometry.from_tensor_4x4(data.rigid)

        for i, block in enumerate(self.blocks):
            s, z, pred_xyz, r = block(data, s, z, r, i)

            preds = {
                "frames": r.to_tensor_4x4(),
                #"sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "positions": pred_xyz,
                #"states": s,
            }

            outputs.append(preds)

            r = r.stop_rot_gradient()

        del z

        outputs = dict_multimap(torch.stack, outputs)

        return outputs

class StructureBlock(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 ipa_no_heads,
                 no_qk_points,
                 no_v_points,
                 block_idx,
                 no_blocks,
                 ):
        super(StructureBlock, self).__init__()
        
        self.no_blocks = no_blocks
        self.edge_ipa = GraphIPA(c_s,
                                 c_hidden,
                                 c_z,
                                 ipa_no_heads,
                                 no_qk_points,
                                 no_v_points,
        )
        self.ipa_ln = LayerNorm(c_s)

        '''
        self.skip_embed = nn.Linear(
            self._model_conf.node_embed_size,
            self._ipa_conf.c_skip,
            init="final"
        )
        '''

        self.node_transition = TransitionLayer(c_s)

        if block_idx < (no_blocks-1):
            self.edge_transition = EdgeTransition(c_s,
                                                c_z,
                                                c_z)

        self.Rigid_update = RigidUpdate(c_s)

    def forward(self, data, s, z, r, i):

        # [N, C_hidden]
        s = s + self.edge_ipa(data, s, z)
        s = self.ipa_ln(s)
        s = self.node_transition(s)
        if i < (self.no_blocks -1):
            z = self.edge_transition(data, s, z)

        r = r.compose_q_update_vec(self.Rigid_update(s, data.bb_mask, data.atom_mask))

        # change to rotation matrix
        r = geometry.Rigid(
            geometry.Rotation(
                rot_mats=r.rot.get_rot_mat(),
                quats=None
            ),
            r.trans,
        )

        identity = geometry.Rigid.identity((len(data.aatype),5), device= r.device)
        flatten_identity = geometry.flatten_rigid(identity)
        t_identity = flatten_identity.to_tensor_4x4()
        t_identity[data.rigid_mask] = r.to_tensor_4x4()

        sidechain_frame_tensor = geometry.unflatten_toN4(t_identity)

        pred_xyz = structure_build.frame_to_14pos(
            sidechain_frame_tensor,
            data.gt_global_frame[...,:4,:,:],
            data.aatype,
            data.bb_coord
        )

        return s, z, pred_xyz, r

class RigidPacking(nn.Module):
    """
    Our Model
    """

    def __init__(self,
                 num_blocks: int = 6, # StructureUpdateModule的循环次数

                 # InputEmbedder config
                 nf_dim: int = 7 + 19,
                 c_s: int = 320, # Node channel dimension after InputEmbedding

                 # PairEmbedder parameter
                 pair_dim: int = 2*32+1 , # rbf+3+4 + nf_dim* 2 + 2* relpos_k+1 + 10 edge type
                 c_z: int = 64, # Pair channel dimension after InputEmbedding
                 #c_hidden_tri_att: int = 16, # x2 cause we x2 the input dimension
                 #c_hidden_tri_mul: int = 32, # Keep ori
                 #pairemb_no_blocks: int = 2, # Keep ori
                 #mha_no_heads: int = 4, # Keep ori
                 #pair_transition_n: int = 2, # Keep ori

                 # IPA config
                 c_hidden: int = 12,  # IPA hidden channel dimension
                 ipa_no_heads: int = 8,  # Number of attention head
                 no_qk_points: int =4,  # Number of qurry/key (3D vector) point head
                 no_v_points: int =8,  # Number of value (3D vector) point head

                 # AngleResnet
                 c_resnet: int = 128, # AngleResnet hidden channel dimension
                 no_resnet_blocks: int = 4, # Resnet block number
                 no_angles: int = 4, # predict chi 1-4 4 angles
                 epsilon: int = 1e-7,
                 top_k: int =64,

                 # Arc config
                 all_loc = False,
                 ):

        super(RigidPacking, self).__init__()

        self.all_loc = all_loc
        self.num_blocks = num_blocks
        self.top_k = top_k
        
        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()

        self.input_embedder = InputEmbedder(nf_dim, c_s,
                                            pair_dim, c_z, # Pair feature related dim
                                            #c_hidden_tri_att, c_hidden_tri_mul, # hidden dim for TriangleAttention, TriangleMultiplication
                                            #pairemb_no_blocks, mha_no_heads, pair_transition_n,
                                            )

        self.structure_update = StructureUpdateModule(num_blocks,
                                                     c_s,
                                                     c_z,
                                                     c_hidden,
                                                     ipa_no_heads,
                                                     no_qk_points,
                                                     no_v_points)

    def forward(self,
                data):
        
        torch.cuda.empty_cache()
        # [N_rigid, (c_s,c_v)], [E, (e_s,e_v)]
        s, z = self.input_embedder(data)

        # [N_rigid, (c_s,c_v)]
        outputs = self.structure_update(data, s, z)

        return outputs

    @classmethod
    def from_dir(
        cls,
        dirname: str,
        load_weights: bool = True,
        best_by: Literal["train", "valid"] = "valid",
        copy_to: str = "",
        **kwargs,
    ):
        """
        Builds this model out from directory. Legacy mode is for loading models
        before there were separate folders for training and validation best models.
        idx indicates which model to load if multiple are given
        """
        train_args_fname = os.path.join(dirname, "training_args.json")
        with open(train_args_fname, "r") as source:
            train_args = json.load(source)

        if load_weights:
            subfolder = f"best_by_{best_by}"
            ckpt_names = glob.glob(os.path.join(dirname, "models", subfolder, "*.ckpt"))
            logging.info(f"Found {len(ckpt_names)} checkpoints")
            ckpt_name = ckpt_names[1] # choose which check point
            logging.info(f"Loading weights from {ckpt_name}")

            retval = cls()
            loaded = torch.load(ckpt_name, map_location=torch.device("cuda"))
            retval.load_state_dict(loaded["state_dict"])

        # If specified, copy out the requisite files to the given directory
        if copy_to:
            logging.info(f"Copying minimal model file set to: {copy_to}")
            os.makedirs(copy_to, exist_ok=True)
            copy_to = Path(copy_to)
            with open(copy_to / "training_args.json", "w") as sink:
                json.dump(train_args, sink)
            if load_weights:
                # Create the direcotry structure
                ckpt_dir = copy_to / "models" / subfolder
                os.makedirs(ckpt_dir, exist_ok=True)
                shutil.copyfile(ckpt_name, ckpt_dir / os.path.basename(ckpt_name))

        return retval

