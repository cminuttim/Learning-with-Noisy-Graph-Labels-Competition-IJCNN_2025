# source/data_loader.py
import gzip
import json
import os
import pandas as pd
from torch_geometric.data import Data, DataLoader
import torch
from sklearn.model_selection import train_test_split

def load_dataset(file_path: str) -> pd.DataFrame:
    data = []
    db = []    
    for file in file_path.split(' '):
        x = os.path.basename(os.path.dirname(file))
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            tmp = json.load(f)
            data = data + tmp
            db = db + [x]*len(tmp)
    data = pd.DataFrame(data)
    data = data.assign(db=db)
    return data

def create_data_loader(df: pd.DataFrame, batch_size: int, train: bool = True) -> DataLoader:
    dataset = create_dataset_from_dataframe(df, result=train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def create_dataset_from_dataframe(df, result=True):
    dataset = []
    for _, row in df.iterrows():
        edge_index = torch.tensor(row['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(row['edge_attr'], dtype=torch.float)  # Ensure 7 features per edge
        num_nodes = row['num_nodes']
        y = torch.tensor([row['y'][0][0]], dtype=torch.long) if result else torch.tensor([0], dtype=torch.long)
        #print(y)

        # Create a Data object
        data = Data(x=torch.ones((num_nodes, 1)),  # Node features (1 feature per node)
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y)
        data.x = torch.nan_to_num(data.x, nan=0.0)
        data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0)

        dataset.append(data)
    return dataset
