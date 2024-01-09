import pickle

import torch
from torch_geometric.data import lightning
from dataset import protein_to_graph
import dataset
from rigid_predict.model.model import RigidPacking
from rigid_predict.model import losses
with open('D:\ProteinProject\RigidPridict\Test_data.pkl', "rb") as file:
    proteins_list = pickle.load(file)

graph_data = dataset.preprocess_datapoints(raw_dir='D:\ProteinProject\RigidPridict\Test_data.pkl')


dset = dataset.ProteinDataset(data=graph_data)

train_set = dset[:5]
validation_set = dset

datamodule = lightning.LightningDataset(train_dataset=train_set,
                                        val_dataset=validation_set,
                                        batch_size=1,
                                        pin_memory=False,
                                        # num_workers=num_workers,
                                        # persistent_workers=True,
                                        follow_batch=['x', 'gt_14pos'])

model = RigidPacking()
for b in datamodule.train_dataloader():

    output = model.forward(b)

    avg_loss = losses.fape_loss(output, b)


