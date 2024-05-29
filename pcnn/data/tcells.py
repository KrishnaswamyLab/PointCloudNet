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
import pandas as pd
import pickle

import torch_geometric
from torch_geometric.io import read_off
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.data import InMemoryDataset, Data

from torch.utils.data import Subset, DataLoader

from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold, StratifiedShuffleSplit

from pcnn import DATA_DIR

import pytorch_lightning as pl
from pcnn.data.utils import laplacian_collate_fn, get_pretransforms
from torch_geometric.data.dataset import _repr, files_exist
from torch_geometric.data.makedirs import makedirs

from sklearn.decomposition import PCA

import sys
import scprep
import pandas as pd
import pickle



class TCellsDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str = 'tcells',
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


        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)
        
        self.train_idx = np.load(os.path.join(self.raw_dir,'train_idx.npy'))
        self.val_idx = np.load(os.path.join(self.raw_dir,'val_idx.npy'))
        self.test_idx = np.load(os.path.join(self.raw_dir,'test_idx.npy'))

        self._patient_labels = None

    @property
    def raw_file_names(self) -> List[str]:
        return [
            f'perturb_labels.npy', f'patient_list.pkl', f'pc.pkl', 'train_idx.npy', 'val_idx.npy', 'test_idx.npy'
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.graph_type}.pt']
    
    @property
    def patient_labels(self):
        if self._patient_labels is not None:
            return self._patient_labels
        else:        
            self._patient_labels = self.data.pat_id
            return self._patient_labels
        
    @property
    def patient_classes(self):
        return self.data.y
    
    @property
    def input_dim(self):
        if self.data.x is not None:
            return self.data.x.shape[-1]
        else:
            return self.data.pos.shape[-1]
        
    @property
    def pos_dim(self):
        return self.input_dim
        
    @property
    def num_classes(self):
        return len(np.unique(self.data.y))
    
    def download(self):

        data_path = os.path.join(DATA_DIR,'perturb_classif.csv')
        data = pd.read_csv(data_path, index_col='PERTURB')

        ### log transform and normalize
        #data_norm = scprep.transform.log(data) #default base=10
        #data_norm = scprep.normalize.library_size_normalize(data_norm, rescale = 1000)

        ##Checking that the list of features is up to date

        df_train_list = []
        df_val_list = []
        df_test_list = []
        for state in np.unique(data.index):
            df_ = data.loc[state]
            train_idx, val_idx = train_test_split(np.arange(len(df_)), test_size=0.4, random_state=42)
            val_idx, test_idx = train_test_split(val_idx, test_size=0.5, random_state=42)
            df_train_ = df_.iloc[train_idx]
            df_val_ = df_.iloc[val_idx]
            df_test_ = df_.iloc[test_idx]
            df_train_list.append(df_train_)
            df_val_list.append(df_val_)
            df_test_list.append(df_test_)
        df_train = pd.concat(df_train_list)
        df_val = pd.concat(df_val_list)
        df_test = pd.concat(df_test_list)
        
        #Subsample the point clouds
        n_points = 200 
        n_samples = (200,50,50) #train, val, test
        
        meta_labels = []
        subsampled_pcs = []
        subsampled_patient_ids = []
        
        for i, (data_,split) in enumerate([(df_train,"train"), (df_val,"val"), (df_test,"test")]):
            labels = []

            label_dict = {"Precursor": 0,
                        "Proliferation": 1,
                        "Interferon": 2,
                        "Stem": 3,
                        "Effector": 4,
                        "Memory": 5,
                        "Exhausted": 6}
            label_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}

            for pat_id in np.unique(list(data_.index)):
                len_pat = data_.loc[pat_id].shape[0]

                for n in range(n_samples[i]):
                    idx = np.random.choice(np.arange(len_pat),size=n_points, replace = True)
                    subsampled_pcs.append(data_.loc[pat_id].iloc[idx].values)
                    subsampled_patient_ids.append(pat_id)

                    label_ = label_dict[pat_id]
                    labels.append(label_)

            labels = np.array(labels)
        
            meta_labels.append(labels)

        train_idx = np.arange(len(meta_labels[0]))
        val_idx = np.arange(len(meta_labels[1])) + len(meta_labels[0])
        test_idx = np.arange(len(meta_labels[2])) + len(meta_labels[0]) + len(meta_labels[1])

        labels = np.concatenate(meta_labels)
        import pickle

        with open(os.path.join(self.raw_dir,f'pc.pkl'), 'wb') as handle:
            pickle.dump(subsampled_pcs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.raw_dir,f'patient_list.pkl'), 'wb') as handle:
            pickle.dump(subsampled_patient_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

        np.save(os.path.join(self.raw_dir,f'perturb_labels.npy'), labels)
        
        np.save(os.path.join(self.raw_dir,f'train_idx.npy'), train_idx)
        np.save(os.path.join(self.raw_dir,f'val_idx.npy'), val_idx)
        np.save(os.path.join(self.raw_dir,f'test_idx.npy'), test_idx)

        return


    def process(self):
        torch.save(self.process_set(), self.processed_paths[0])

    def process_set(self):

        with open(os.path.join(self.raw_dir,'pc.pkl'), 'rb') as handle:
            subsampled_pcs = pickle.load(handle)

        with open(os.path.join(self.raw_dir,'patient_list.pkl'), 'rb') as handle:
            subsampled_patient_ids = pickle.load(handle)

        labels = np.load(os.path.join(self.raw_dir,'perturb_labels.npy'))

        data_list = []
        for i in range(len(subsampled_pcs)):
            d_ = Data(y=torch.tensor([labels[i]]).long(),
                         pos=torch.tensor(subsampled_pcs[i]).float(), pat_id=subsampled_patient_ids[i])
            data_list.append(d_)

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
                    #data_list = p.map(self.pre_transform, data_list)

            if self.normalize_scattering_features:
                if hasattr(data_list[0],"scattering_features"):
                    print("Normalizing scattering features....")
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


class TCellsSelectTransform(object):
    def __init__(self, select_feats, variant):
        
        if variant == "myeloid":
            self.all_feats = MYELOID_FEATS
        self.select_feats = select_feats
        
        ## Selecting the features to use.
        self.idx = [np.where(np.array(self.all_feats) == feat)[0][0] for feat in self.select_feats]

    def __call__(self,data):
        data.x = data.pos[:,self.idx]
        return data
    
class DummyTCellsSelectTransform(object):
    def __init__(self):
        return

    def __call__(self,data):
        data.x = data.pos
        return data


class TCellsData(pl.LightningDataModule):
    def __init__(self, 
                 n_samples = 100, 
                 batch_size = 32, 
                 num_workers = 4, 
                 pin_memory = True, 
                 random_state = 42, 
                 re_precompute = True, 
                 njobs = 1, 
                 reprocess_if_different = True, 
                 train_size = None,
                 **kwargs):

        """
        k: number of nearest neighbors to consider
        n_samples: number of samples for each point cloud
        """

        super().__init__()
        self.save_hyperparameters()
        dataset_alias = f"tcells" #@param ["ModelNet10", "ModelNet40"] {type:"raw"}
        #self.feats = feats

        
        if re_precompute:
            if os.path.isdir(os.path.join(DATA_DIR,dataset_alias,"processed")):
                shutil.rmtree(os.path.join(DATA_DIR,dataset_alias,"processed"))
        
        if "scattering_n_pca" in kwargs:
            scattering_n_pca = kwargs["scattering_n_pca"]
        else:
            scattering_n_pca = None

        graph_type = kwargs["graph_construct"]["graph_type"]

        self.collate_fn = laplacian_collate_fn
        
        base_pre_transform = [T.NormalizeScale(), DummyTCellsSelectTransform()]
        pre_transform_list = get_pretransforms(pre_transforms_base = base_pre_transform, scattering_n_pca = scattering_n_pca, **kwargs["graph_construct"])
        pre_transform = T.Compose(pre_transform_list)

        transform = None #
        dataset = TCellsDataset(
            root= os.path.join(DATA_DIR,dataset_alias),
            transform=transform,
            pre_transform=pre_transform,
            njobs = njobs,
            reprocess_if_different = reprocess_if_different,
            normalize_scattering_features= kwargs["graph_construct"]["normalize_scattering_features"],
            scattering_n_pca=scattering_n_pca,
            graph_type=graph_type
        )

        #gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=random_state)
        #gss_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
        

        self.train_dataset = Subset(dataset, dataset.train_idx)
        self.val_dataset = Subset(dataset, dataset.val_idx)
        self.test_dataset = Subset(dataset, dataset.test_idx)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.input_dim = dataset.input_dim
        self.num_classes = dataset.num_classes
        self.pos_dim = dataset.pos_dim

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