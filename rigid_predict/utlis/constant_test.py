"""Frame init positions for each residue,
   Central atom position are set to [0,0,0],
   Most atoms are located in the xy plane initially,
   ATOM frame are set to a single group, with Initial
   coordinate set to [0,0,0] """
'''
CG position is defined so that CB is the origin, CG - CB, the rotational axis, as the x axis, 
CG init location is in the xy-plane so that the init chi angle is 0.
'''
'''we did NOT change the x axis for convenient'''

import numpy as np
from rigid_predict.utlis import constant

residues_atom_position = {
    'ALA': [
        ['N', 0, (-0.525, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.529, -0.774, -1.205)],
        ['O', 0, (0.627, 1.062, 0.000)],
    ],
    'ARG': [
        ['N', 0, (-0.524, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.524, -0.778, -1.209)],
        ['O', 0, (0.626, 1.062, 0.000)],

        ['CG', 1, (0.000, 0.000, -0.000)],
        ['CD', 1, (1.522, 0.000, 0.000)],

        ['NE', 2, (0.000, 0.000, -0.000)],
        ['NH1', 2, (0.206, 2.301, 0.000)],
        ['NH2', 2, (2.078, 0.978, -0.000)],
        ['CZ', 2, (0.758, 1.093, -0.000)],
    ],
    'ASN': [
        ['N', 0, (-0.536, 1.357, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.531, -0.787, -1.200)],
        ['O', 0, (0.625, 1.062, 0.000)],

        ['CG', 1, (0.000, 0.000, 0.000)],
        ['ND2', 1, (0.593, -1.188, 0.001)],
        ['OD1', 1, (0.633, 1.059, 0.000)],
    ],
    'ASP': [
        ['N', 0, (-0.525, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, 0.000, -0.000)],
        ['CB', 0, (-0.526, -0.778, -1.208)],
        ['O', 0, (0.626, 1.062, -0.000)],

        ['CG', 1, (0.000, 0.000, -0.000)],
        ['OD1', 1, (0.610, 1.091, 0.000)],
        ['OD2', 1, (0.592, -1.101, -0.003)],
    ],
    'CYS': [
        ['N', 0, (-0.522, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, 0.000)],
        ['CB', 0, (-0.519, -0.773, -1.212)],
        ['O', 0, (0.625, 1.062, -0.000)],

        ['SG', 1, (0.000, 0.000, 0.000)],  # Atom node init coordinate are all set to 0.000
    ],
    'GLN': [
        ['N', 0, (-0.526, 1.361, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.779, -1.207)],
        ['O', 0, (0.626, 1.062, -0.000)],

        ['CG', 1, (0.000, 0.000, 0.000)],

        ['CD', 2, (0.000, 0.000, -0.000)],
        ['NE2', 2, (0.593, -1.189, -0.001)],
        ['OE1', 2, (0.634, 1.060, 0.000)],
    ],
    'GLU': [
        ['N', 0, (-0.528, 1.361, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.526, -0.781, -1.207)],
        ['O', 0, (0.626, 1.062, 0.000)],

        ['CG', 1, (0.000, 0.000, 0.000)],

        ['CD', 2, (0.000, 0.000, 0.000)],
        ['OE1', 2, (0.607, 1.095, -0.000)],
        ['OE2', 2, (0.589, -1.104, -0.001)],
    ],
    'GLY': [
        ['N', 0, (-0.572, 1.337, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.517, -0.000, -0.000)],
        ['O', 0, (0.626, 1.062, -0.000)],
    ],
    'HIS': [
        ['N', 0, (-0.527, 1.360, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.778, -1.208)],
        ['O', 0, (0.625, 1.063, 0.000)],

        ['CG', 1, (0.000, 0.000, -0.000)],
        ['CD2', 1, (0.889, -1.021, 0.003)],
        ['ND1', 1, (0.744, 1.160, -0.000)],
        ['CE1', 1, (2.030, 0.851, 0.002)],
        ['NE2', 1, (2.145, -0.466, 0.004)],
    ],
    'ILE': [
        ['N', 0, (-0.493, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.536, -0.793, -1.213)],
        ['O', 0, (0.627, 1.062, -0.000)],
        # Before: CB as origin
        # Now: CG1 as origin
        ['CG1', 1, (0.000, 0.000, -0.000)],
        ['CG2', 1, (-2.081, 1.430, -0.000)],
        #
        ['CD1', 2, (0.000, 0.000, 0.000)],
    ],
    'LEU': [
        ['N', 0, (-0.520, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.773, -1.214)],
        ['O', 0, (0.625, 1.063, -0.000)],

        ['CG', 1, (0.000, 0.000, 0.000)],
        ['CD1', 1, (0.530, 1.430, -0.000)],
        ['CD2', 1, (0.535, -0.774, 1.200)],
    ],
    'LYS': [
        ['N', 0, (-0.526, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.524, -0.778, -1.208)],
        ['O', 0, (0.626, 1.062, -0.000)],

        ['CG', 1, (0.000, 0.000, 0.000)],
        ['CD', 1, (0.559, 1.417, 0.000)],

        ['CE', 2, (0.000, 0.000, 0.000)],
        ['NZ', 2, (0.554, 1.387, 0.000)],
    ],
    'MET': [
        ['N', 0, (-0.521, 1.364, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.210)],
        ['O', 0, (0.625, 1.062, -0.000)],

        ['CG', 1, (0.000, 0.000, -0.000)],

        ['SD', 2, (0.000, 0.000, 0.000)],
        ['CE', 2, (0.320, 1.786, -0.000)],
    ],
    'PHE': [
        ['N', 0, (-0.518, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, -0.000)],
        ['CB', 0, (-0.525, -0.776, -1.212)],
        ['O', 0, (0.626, 1.062, -0.000)],

        ['CG', 1, (0.000, 0.000, 0.000)],
        ['CD1', 1, (0.709, 1.195, -0.000)],
        ['CD2', 1, (0.706, -1.196, 0.000)],
        ['CE1', 1, (2.102, 1.198, -0.000)],
        ['CE2', 1, (2.098, -1.201, -0.000)],
        ['CZ', 1, (2.794, -0.003, -0.001)],
    ],
    'PRO': [  # We set all the PRO atom into one rigid group which leave the chi angle constant
        ['N', 0, (-0.566, 1.351, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, 0.000)],
        ['CB', 0, (-0.546, -0.611, -1.293)],
        ['O', 0, (0.621, 1.066, 0.000)],
        ['CG', 0, (-1.833, 0.0971, -1.498)],  # from group 5 to group 0
        ['CD', 0, (-1.663, 1.492, -0.962)],  # from group 5 to group 0
        # ['CG', 4, (0.382, 1.445, 0.0)], # from group 5 to group 0
        # ['CD', 0, (0.427, 1.440, 0.0)], # from group 5 to group 0, using the correct angle.
        # ['CD', 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    'SER': [
        ['N', 0, (-0.529, 1.360, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.518, -0.777, -1.211)],
        ['O', 0, (0.626, 1.062, -0.000)],

        ['OG', 1, (0.000, 0.000, 0.000)],
    ],
    'THR': [
        ['N', 0, (-0.517, 1.364, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, -0.000)],
        ['CB', 0, (-0.516, -0.793, -1.215)],
        ['O', 0, (0.626, 1.062, 0.000)],

        ['CG2', 1, (0.000, 0.000, 0.000)],
        ['OG1', 1, (-1.930, 1.442, 0.000)],
    ],
    'TRP': [
        ['N', 0, (-0.521, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.212)],
        ['O', 0, (0.627, 1.062, 0.000)],

        ['CG', 1, (0.000, 0.000, -0.000)],
        ['CD1', 1, (0.824, 1.091, 0.000)],
        ['CD2', 1, (0.854, -1.148, -0.005)],
        ['CE2', 1, (2.186, -0.678, -0.007)],
        ['CE3', 1, (0.622, -2.530, -0.007)],
        ['NE1', 1, (2.140, 0.690, -0.004)],
        ['CH2', 1, (3.028, -2.890, -0.013)],
        ['CZ2', 1, (3.283, -1.543, -0.011)],
        ['CZ3', 1, (1.715, -3.389, -0.011)],
    ],
    'TYR': [
        ['N', 0, (-0.522, 1.362, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.776, -1.213)],
        ['O', 0, (0.627, 1.062, -0.000)],

        ['CG', 1, (0.000, 0.000, -0.000)],
        ['CD1', 1, (0.716, 1.195, -0.000)],
        ['CD2', 1, (0.713, -1.194, -0.001)],
        ['CE1', 1, (2.107, 1.200, -0.002)],
        ['CE2', 1, (2.104, -1.201, -0.003)],
        ['OH', 1, (4.168, -0.002, -0.005)],
        ['CZ', 1, (2.791, -0.001, -0.003)],
    ],
    'VAL': [
        ['N', 0, (-0.494, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.533, -0.795, -1.213)],
        ['O', 0, (0.627, 1.062, -0.000)],

        ['CG1', 1, (0.000, 0.000, -0.000)],
        ['CG2', 1, (0.533 - 0.540, -0.776 - 1.429, 1.203)],
    ],
    'UNK': [  # Adding Atom for UNK residues, as UNK can be different res, here
        # I am naively using ALA's atom position. Notice that this can be wrong
        # when we Modify the Main chain position
        # 算主链的时候要注意这个，UNK 用不同的氨基酸原子位置可能导致其他res的位置不准确
        ['N', 0, (-0.525, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.529, -0.774, -1.205)],
        ['O', 0, (0.627, 1.062, 0.000)],
    ],
}

'[ -x axis atom, origin, xy-plane atom ]'
frame_atoms = {
    'ALA': [],
    'ARG': [['CB', 'CG', 'CD'], ['CD', 'NE', 'CZ']],
    'ASN': [['CB', 'CG', 'OD1']],
    'ASP': [['CB', 'CG', 'OD1']],
    'CYS': [['', 'SG', '']],  # POINT HAS NO FRAME
    'GLN': [['', 'CG', ''], ['CG', 'CD', 'OE1']],
    'GLU': [['', 'CG', ''], ['CG', 'CD', 'OE1']],
    'GLY': [],
    'HIS': [['CB', 'CG', 'ND1']],
    'ILE': [['CB', 'CG1', 'CG2'], ['', 'CD1', '']],
    'LEU': [['CB', 'CG', 'CD1']],
    'LYS': [['CB', 'CG', 'CD'], ['CD', 'CE', 'NZ']],
    'MET': [['', 'CG', ''], ['CG', 'SD', 'CE']],
    'PHE': [['CB', 'CG', 'CD1']],
    'PRO': [],
    'SER': [['', 'OG', '']],
    'THR': [['CB', 'OG1', 'CG2']],
    'TRP': [['CB', 'CG', 'CD1']],
    'TYR': [['CB', 'CG', 'CD1']],
    'VAL': [['CB', 'CG1', 'CG2']],
    'UNK': [],
}
frame_mask = [
    [1.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0],  # ASN
    [1.0, 1.0, 0.0],  # ASP
    [1.0, 1.0, 0.0],  # CYS
    [1.0, 1.0, 1.0],  # GLN
    [1.0, 1.0, 1.0],  # GLU
    [1.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0],  # HIS
    [1.0, 1.0, 1.0],  # ILE
    [1.0, 1.0, 0.0],  # LEU
    [1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0],  # MET
    [1.0, 1.0, 0.0],  # PHE
    [1.0, 0.0, 0.0],  # PRO
    [1.0, 1.0, 0.0],  # SER
    [1.0, 1.0, 0.0],  # THR
    [1.0, 1.0, 0.0],  # TRP
    [1.0, 1.0, 0.0],  # TYR
    [1.0, 1.0, 0.0],  # VAL
    [1.0, 0.0, 0.0],  # UNK 暂且先不给UNK 安排任何chi angle
]

middle_atom_mask = [
    [0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0],  # ASN
    [0.0, 0.0, 0.0],  # ASP
    [0.0, 1.0, 0.0],  # CYS
    [0.0, 1.0, 0.0],  # GLN
    [0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 1.0],  # ILE
    [0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0],  # LYS
    [0.0, 1.0, 0.0],  # MET
    [0.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0],  # PRO
    [0.0, 1.0, 0.0],  # SER
    [0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0],  # TRP
    [0.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0],  # UNK 暂且先不给UNK 安排任何chi angle
]

chi_angles_atoms = {
    'ALA': [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLY': [],
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'PRO': [],  # ['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'VAL': [['N', 'CA', 'CB', 'CG1']],
    'UNK': [],
}

atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]

atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}


restype_name_to_rigid_idx = {
    "ALA": [2],
    "ARG": [2, 8, 10],  # psp13 [1,8,8,19,20]
    "ASN": [2, 7],  # psp14
    "ASP": [2, 6],  # psp11
    "CYS": [2, 18],  # psp15
    "GLN": [2, 0, 7],  # psp14
    "GLU": [2, 19, 6],  # psp11
    "GLY": [1],
    "HIS": [2, 15],  # psp17
    "ILE": [2, 3, 0],  # not show in psp or group 6? *****# [1,9,10]
    "LEU": [2, 4],  # psp12
    "LYS": [2, 8, 9],  # psp10
    "MET": [2, 19, 11],  # psp9 # [1,8,8,11]
    "PHE": [2, 12],  # psp16
    "PRO": [16],  # psp19
    "SER": [2, 17],  # psp2
    "THR": [2, 5],  # psp15
    "TRP": [2, 14],  # psp18
    "TYR": [2, 13],  # not show in psp *********
    "VAL": [2, 3],  # psp12
    "UNK": [1],  # 给 UNKNOWN res 只添加主链 rigid 其他原子先不管
}


restype_atom14_to_rigid_group = np.zeros([21, 14], dtype=int)
restype_atom14_mask = np.zeros([21, 14], dtype=np.float32)
restype_atom14_rigid_group_positions = np.zeros([21, 14, 3], dtype=np.float32)
restype_rigid_group_default_frame = np.zeros([21, 3, 4, 4], dtype=np.float32)
restype_rigidtype = np.zeros([21, 3], dtype=int)
restype_rigid_mask = np.zeros([21, 3], dtype=np.float32)
def _make_rigid_group_constants():
    """Make rigid frames signed to every 14atoms for each residue"""

    for residx, restype1 in enumerate(constant.restypes):
        restype3 = constant.restype_1to3[restype1]

        for name, group_idx, init_atom_pos in residues_atom_position[restype3]:
            atom14idx = constant.restype_name_to_atom14_names[restype3].index(name)
            restype_atom14_to_rigid_group[residx, atom14idx] = group_idx
            restype_atom14_mask[residx, atom14idx] = 1
            restype_atom14_rigid_group_positions[residx, atom14idx, :] = init_atom_pos

        restype_rigid_group_default_frame[residx, 0, :, :] = np.eye(4)
        for i, rigid_type in enumerate(restype_name_to_rigid_idx[restype3]):
            restype_rigidtype[residx, i] = rigid_type
            restype_rigid_mask[residx, i] = rigid_type


_make_rigid_group_constants()


