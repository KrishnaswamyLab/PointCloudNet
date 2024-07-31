import os
import glob
import os.path as osp
import torch
import torch.nn.functional as F

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import shutil
import tqdm

import torch_geometric
from torch_geometric.io import read_off
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet

from torch.utils.data import Subset, DataLoader

from sklearn.model_selection import train_test_split

from pcnn import DATA_DIR

import pytorch_lightning as pl
from pcnn.data.utils import laplacian_collate_fn, get_pretransforms
from torch_geometric.data.dataset import _repr, files_exist
from torch_geometric.data.makedirs import makedirs

from sklearn.decomposition import PCA

import sys


class ModelNetExt(ModelNet):
    def __init__(
        self,
        root: str,
        name: str = '10',
        train: bool = True,
        transform = None,
        pre_transform = None,
        pre_filter = None,
        njobs = 1,
        normalize_scattering_features = True,
        reprocess_if_different = True,
        scattering_n_pca = None,
        graph_type = "knn",
    ):
        self.njobs = njobs
        self.normalize_scattering_features = normalize_scattering_features
        self.reprocess_if_different = reprocess_if_different
        self.scattering_n_pca = scattering_n_pca
        self.graph_type = graph_type
        super().__init__(root, name, train, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.graph_type}_training.pt', f'{self.graph_type}_test.pt']
        
    def process_set(self,dataset):
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        data_list = []
        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob(f'{folder}/{category}_*.off')
            for path in paths:
                data = read_off(path)
                data.y = torch.tensor([target])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            if self.njobs==1:
                d_list = []
                for d in tqdm.tqdm(data_list):
                    d_list.append(self.pre_transform(d))
                data_list = d_list
            else:
                from pathos.multiprocessing import ProcessingPool as Pool
                with Pool(self.njobs) as p:
                    data_list = list(tqdm.tqdm(p.imap(self.pre_transform, data_list), total=len(data_list)))

            if self.normalize_scattering_features:
                print("Normalizing scattering features....")
                if hasattr(data_list[0],"scattering_features"):
                    scat_feats = []
                    d_list = []
                    for d in data_list:
                        scat_feats.append(d.scattering_features)
                    scat_feats = torch.cat(scat_feats)

                    if self.scattering_n_pca is not None:
                        pca = PCA(n_components=self.scattering_n_pca)
                        scat_feats = pca.fit_transform(scat_feats.reshape(scat_feats.shape[0],-1))
                        scat_feats = torch.tensor(scat_feats).float()

                        max_scat = scat_feats[~torch.isinf(scat_feats)].max() 
                        
                        for d in data_list:
                            d.scattering_features = torch.Tensor(pca.transform(d.scattering_features.reshape(1,-1)))
                            d.scattering_features[torch.isinf(d.scattering_features)] = max_scat
                            d_list.append(d) 

                    else:
                        m_scat = scat_feats.mean(0)[None,...]
                        std_scat = scat_feats.std(0)[None,...]
                        max_scat = scat_feats[~torch.isinf(scat_feats)].max()

                        for d in data_list:
                            d.scattering_features = d.scattering_features - m_scat / std_scat
                            d.scattering_features[torch.isinf(d.scattering_features)] = max_scat
                            d_list.append(d) 
                    data_list = d_list

        return self.collate(data_list)
    
    def _process(self):
        f = osp.join(self.processed_dir, f'{self.graph_type}_pre_transform.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            print(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first")
            if self.reprocess_if_different:
                print("Reprocessing dataset...")
                files = os.listdir(self.processed_dir)
                files_to_delete = [f_ for f_ in files if self.graph_type in f]
                for f_ in files_to_delete:
                    os.remove(os.path.join(self.processed_dir,f_))

        f = osp.join(self.processed_dir, f'{self.graph_type}_pre_filter.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            print(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, make sure to "
                "delete '{self.processed_dir}' first")

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        if self.log and 'pytest' not in sys.modules:
            print('Processing...', file=sys.stderr)

        makedirs(self.processed_dir)
        self.process()

        path = osp.join(self.processed_dir, f'{self.graph_type}_pre_transform.pt')
        torch.save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, f'{self.graph_type}_pre_filter.pt')
        torch.save(_repr(self.pre_filter), path)

        if self.log and 'pytest' not in sys.modules:
            print('Done!', file=sys.stderr)


class ModelNetData(pl.LightningDataModule):
    def __init__(self, n_samples = 100, batch_size = 32, num_workers = 4, pin_memory = True, random_state = 42, re_precompute = True, njobs = 1, reprocess_if_different = True, train_size = None, **kwargs):
        """
        k: number of nearest neighbors to consider
        n_samples: number of samples for each point cloud
        """
        super().__init__()
        self.save_hyperparameters()
        display_sample = n_samples  #@param {type:"slider", min:256, max:4096, step:16}
        modelnet_dataset_alias = "ModelNet10" #@param ["ModelNet10", "ModelNet40"] {type:"raw"}

        # Classes for ModelNet10 and ModelNet40
        categories = sorted([
            x.split(os.sep)[-2]
            for x in glob.glob(os.path.join(
                modelnet_dataset_alias, "raw", '*', ''
            ))
        ])

        def signal_transform(x):
            if x.x is None:
                x.x = x.pos
            return x
        
        if re_precompute:
            if os.path.isdir(os.path.join(DATA_DIR,modelnet_dataset_alias,"processed")):
                shutil.rmtree(os.path.join(DATA_DIR,modelnet_dataset_alias,"processed"))
        
        if "scattering_n_pca" in kwargs:
            scattering_n_pca = kwargs["scattering_n_pca"]
        else:
            scattering_n_pca = None

        graph_type = kwargs["graph_construct"]["graph_type"]

        self.collate_fn = laplacian_collate_fn
        
        base_pre_transform = [T.NormalizeScale(), T.SamplePoints(display_sample) ]
        pre_transform_list = get_pretransforms(pre_transforms_base = base_pre_transform, scattering_n_pca = scattering_n_pca, **kwargs["graph_construct"])
        pre_transform = T.Compose(pre_transform_list)

        transform = T.Compose([signal_transform]) # setting the signal as the position of the points
        train_dataset = ModelNetExt(
            root= os.path.join(DATA_DIR,modelnet_dataset_alias),
            name=modelnet_dataset_alias[-2:],
            train=True,
            transform=transform,
            pre_transform=pre_transform,
            njobs = njobs,
            reprocess_if_different = reprocess_if_different,
            normalize_scattering_features= kwargs["graph_construct"]["normalize_scattering_features"],
            scattering_n_pca=scattering_n_pca,
            graph_type=graph_type
        )

        self.test_dataset = ModelNetExt(
            root=os.path.join(DATA_DIR,modelnet_dataset_alias),
            name=modelnet_dataset_alias[-2:],
            train=False,
            transform=transform,
            pre_transform=pre_transform,
            njobs = njobs,
            reprocess_if_different = reprocess_if_different,
            normalize_scattering_features= kwargs["graph_construct"]["normalize_scattering_features"],
            scattering_n_pca=scattering_n_pca,
            graph_type=graph_type
        )
        
        train_idx, val_idx = train_test_split(np.arange(len(train_dataset)), test_size=0.2, random_state=random_state)

        if train_size is not None:
            train_idx = train_idx[:train_size]

        self.train_dataset = Subset(train_dataset, train_idx)
        self.val_dataset = Subset(train_dataset, val_idx)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.input_dim = 3
        self.num_classes = 10
        self.pos_dim = 3

    def prepare_data(self):
        pass

    def setup(self, stage = None):
        pass

    def teardown(self, stage = None):
        pass
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,
                          collate_fn = self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,
                          shuffle = False,
                          collate_fn = self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,
                          shuffle = False,
                          collate_fn = self.collate_fn)