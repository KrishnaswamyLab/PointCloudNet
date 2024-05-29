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
from torch_geometric.data import InMemoryDataset, Data

from torch.utils.data import Subset, DataLoader

from sklearn.model_selection import train_test_split, GroupShuffleSplit

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


MELANOMA_FEATS = ['beta-tubulin', 'CD11b', 'CD11c', 'CD14', 'CD163', 'CD20', 'CD3',
       'CD31', 'CD4', 'CD45', 'CD45RO', 'CD56', 'CD68', 'CD8', 'dsDNA',
       'FOXP3', 'Granzyme B', 'HLA class 1 A, B, and C, Na-K-ATPase',
       'HLA DPDQDR', 'IDO-1', 'Ki-67', 'LAG3', 'PD-1', 'PD-L1', 'Podoplanin',
       'SMA', 'SOX10', 'TIM-3', 'Vimentin']

class MelanomaDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str = 'melanoma',
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
        self.feats_names = MELANOMA_FEATS

        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)


        self._patient_labels = None

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'labels.npy', 'patient_list.pkl', 'pc.pkl'
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
    def input_dim(self):
        if self.data.x is not None:
            return self.data.x.shape[-1]
        else:
            return self.data.pos.shape[-1]
        
    @property
    def pos_dim(self):
        return len(MELANOMA_FEATS)
        
    @property
    def num_classes(self):
        return len(np.unique(self.data.y))
    
    def download(self):
        scprep.io.download.download_google_drive("16nbyiNv-AX9zVBHC1XUfN0LWGeA1Sf0k", os.path.join(self.raw_dir,"scaled_cell_intensities.csv"))
        data = pd.read_csv(os.path.join(self.raw_dir,"scaled_cell_intensities.csv"), index_col = 0).drop(columns = ['area', 'x_centroid', 'y_centroid', 'Cell Instance'])
        data.index = [data.index[i].split("_")[1] for i, lab in enumerate(data.index)]

        ### log transform and normalize
        data_norm = scprep.transform.log(data) #default base=10
        data_norm = scprep.normalize.library_size_normalize(data_norm, rescale = 1000)

        ##Checking that the list of features is up to date
        assert (data_norm.columns == self.feats_names).all()

        #unique patient and cell identifiers
        grouped_data = data_norm.groupby(data_norm.index).cumcount() + 1
        patient_data = {p: group for p, group in data_norm.groupby(data_norm.index)}
        metadata = pd.DataFrame()
        metadata["patient_id"] = data_norm.index
        metadata.index = [f"{p}_{i}" for p, i in zip(metadata["patient_id"], grouped_data)]
        for p in patient_data:
            patient_data[p].index = [f"{p}_{i}" for p, i in zip(patient_data[p].index, grouped_data)]

        ##store UMI in metadata
        metadata.index = metadata.index + "_" + grouped_data.astype(str)

        scprep.io.download.download_google_drive("17CHXcF0FKRt-QkEaw84lBvTZ5MqnDytT", os.path.join(self.raw_dir,"melanoma_clinical_info_MIBI.csv"))
        clinical_data = pd.read_csv(os.path.join(self.raw_dir,"melanoma_clinical_info_MIBI.csv"), index_col = 0)

        # patient_ID to match data labels
        c_id = list(clinical_data['376_1_col']) #column on MIBI plate
        r_id = list(clinical_data['376_1_row']) #row on MIBI plate
        clinical_data["Patient_ID"] = np.array(["R"+str(c_id[i])+'C'+str(r_id[i]) for i, idx in enumerate(clinical_data.index) ])\

        #filter by patient IDs with samples present
        res = [i for i, val in enumerate(clinical_data["Patient_ID"]) if val in data_norm.index]
        clinical_data = clinical_data.iloc[res].set_index("Patient_ID")


        #Subsample the point clouds
        n_points = 400
        n_samples = 10
        subsampled_pcs = []
        subsampled_patient_ids = []
        labels = []

        label_dict = {"NO":0,"YES":1}
        for pat_id in patient_data.keys():
            len_pat = len(patient_data[pat_id])
            for n in range(n_samples):
                idx = np.random.choice(np.arange(len_pat),size=n_points, replace = False)
                subsampled_pcs.append(patient_data[pat_id].iloc[idx].values)
                subsampled_patient_ids.append(pat_id)
                
                label_str = clinical_data.loc[pat_id]["RESPONSE"]
                labels.append(label_dict[label_str])

        labels = np.array(labels)

        import pickle

        with open(os.path.join(self.raw_dir,'pc.pkl'), 'wb') as handle:
            pickle.dump(subsampled_pcs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.raw_dir,'patient_list.pkl'), 'wb') as handle:
            pickle.dump(subsampled_patient_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

        np.save(os.path.join(self.raw_dir,'labels.npy'), labels)
        return


    def process(self):
        torch.save(self.process_set(), self.processed_paths[0])

    def process_set(self):

        with open(os.path.join(self.raw_dir,'pc.pkl'), 'rb') as handle:
            subsampled_pcs = pickle.load(handle)

        with open(os.path.join(self.raw_dir,'patient_list.pkl'), 'rb') as handle:
            subsampled_patient_ids = pickle.load(handle)

        labels = np.load(os.path.join(self.raw_dir,'labels.npy'))

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


class MelanomaSelectTransform(object):
    def __init__(self, select_feats):

        self.all_feats = MELANOMA_FEATS
        self.select_feats = select_feats
        
        ## Selecting the features to use.
        self.idx = [np.where(np.array(self.all_feats) == feat)[0][0] for feat in self.select_feats]

    def __call__(self,data):
        data.x = data.pos[:,self.idx]
        return data


class MelanomaData(pl.LightningDataModule):
    def __init__(self, n_samples = 100, 
                 batch_size = 32, 
                 num_workers = 4, 
                 pin_memory = True, 
                 random_state = 42, 
                 re_precompute = True, 
                 njobs = 1, 
                 reprocess_if_different = True, 
                 train_size = None,
                 feats = ["PD-1", "PD-L1", "Ki-67", "LAG3", "TIM-3", "IDO-1", "Granzyme B", "FOXP3", "CD11b", "CD11c", "CD14"],
                  **kwargs):

        """
        k: number of nearest neighbors to consider
        n_samples: number of samples for each point cloud
        """

        super().__init__()
        self.save_hyperparameters()
        modelnet_dataset_alias = "melanoma" #@param ["ModelNet10", "ModelNet40"] {type:"raw"}
        self.feats = feats

        
        if re_precompute:
            if os.path.isdir(os.path.join(DATA_DIR,modelnet_dataset_alias,"processed")):
                shutil.rmtree(os.path.join(DATA_DIR,modelnet_dataset_alias,"processed"))
        
        if "scattering_n_pca" in kwargs:
            scattering_n_pca = kwargs["scattering_n_pca"]
        else:
            scattering_n_pca = None

        graph_type = kwargs["graph_construct"]["graph_type"]

        self.collate_fn = laplacian_collate_fn
        
        base_pre_transform = [T.NormalizeScale(), MelanomaSelectTransform(select_feats = self.feats)]
        pre_transform_list = get_pretransforms(pre_transforms_base = base_pre_transform, scattering_n_pca = scattering_n_pca, **kwargs["graph_construct"])
        pre_transform = T.Compose(pre_transform_list)

        transform = None #
        dataset = MelanomaDataset(
            root= os.path.join(DATA_DIR,modelnet_dataset_alias),
            name=modelnet_dataset_alias[-2:],
            transform=transform,
            pre_transform=pre_transform,
            njobs = njobs,
            reprocess_if_different = reprocess_if_different,
            normalize_scattering_features= kwargs["graph_construct"]["normalize_scattering_features"],
            scattering_n_pca=scattering_n_pca,
            graph_type=graph_type
        )

        gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=random_state)
        gss_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
        train_idx, val_idx = next(gss.split(np.arange(len(dataset)), dataset.patient_labels, groups = dataset.patient_labels))
        val_idx, test_idx = next(gss_test.split(val_idx, np.array(dataset.patient_labels)[val_idx], groups = np.array(dataset.patient_labels)[val_idx]))

        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset = Subset(dataset, val_idx)
        self.test_dataset = Subset(dataset, test_idx)

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