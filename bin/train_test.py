
from rigid_predict.data import dataset
from torch_geometric.data import lightning
from rigid_predict.model.model import RigidPacking
graph_data = dataset.preprocess_datapoints(raw_dir ="/home/PJLAB/zhangyiqiu/PycharmProjects/sidechain-score-v3/unused_file/foldingdiff/test_data.pkl")

dset = dataset.ProteinDataset(data = graph_data)

split_idx = int(len(graph_data) * 0.9)
train_set = dset[:split_idx]
validation_set = dset[split_idx:]

datamodule = lightning.LightningDataset(train_dataset=train_set,
                                        val_dataset=validation_set,
                                        batch_size=2,
                                        pin_memory=False,
                                        #num_workers=num_workers,
                                        #persistent_workers=True,
                                        follow_batch=['x'])

for batch in datamodule.train_dataloader():

    model = RigidPacking()

    out = model.forward(batch)