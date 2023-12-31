from rigid_predict.data import dataset
from torch_geometric.data import lightning
from rigid_predict.model.model import RigidPacking
from rigid_predict.model import losses
graph_data = dataset.preprocess_datapoints(raw_dir='D:\ProteinProject\RigidPridict\Test_data.pkl')

dset = dataset.ProteinDataset(data=graph_data)

split_idx = int(len(graph_data) * 0.1)
train_set = dset[:split_idx]
validation_set = dset[split_idx:]

datamodule = lightning.LightningDataset(train_dataset=train_set,
                                        val_dataset=validation_set,
                                        batch_size=1,
                                        pin_memory=False,
                                        # num_workers=num_workers,
                                        # persistent_workers=True,
                                        follow_batch=['x', 'gt_14pos'])
model = RigidPacking()

for batch in datamodule.train_dataloader():

    outputs = model.forward(batch)

    avg_loss = losses.fape_loss(outputs, batch)