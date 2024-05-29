import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from torchvision import datasets
import gzip

import os
import os.path as osp
from pcnn import DATA_DIR
import torch
import torch_geometric
from torch_geometric.data import  Data
from torch_geometric.data.in_memory_dataset import InMemoryDataset

import shutil
import tqdm

from sklearn.model_selection import train_test_split

import torch_geometric.transforms as T

import pytorch_lightning as pl
from pcnn.data.utils import laplacian_collate_fn, get_pretransforms
from torch_geometric.data.dataset import _repr, files_exist
from torch_geometric.data.makedirs import makedirs

from torch.utils.data import Subset, DataLoader

import sys


def create_raw_data(out_path):
    # choose how many points in point cloud
    N = 2000

    # for projection
    NORTHPOLE_EPSILON = 1e-3


    def meshgrid(b, grid_type='Driscoll-Healy'):
        return np.meshgrid(*linspace(b, grid_type), indexing='ij')

    def linspace(b, grid_type='Driscoll-Healy'):
        if grid_type == 'Driscoll-Healy':
            beta = np.arange(2 * b) * np.pi / (2. * b)
            alpha = np.arange(2 * b) * np.pi / b
        elif grid_type == 'SOFT':
            beta = np.pi * (2 * np.arange(2 * b) + 1) / (4. * b)
            alpha = np.arange(2 * b) * np.pi / b
        elif grid_type == 'Clenshaw-Curtis':
            # beta = np.arange(2 * b + 1) * np.pi / (2 * b)
            # alpha = np.arange(2 * b + 2) * np.pi / (b + 1)
            # Must use np.linspace to prevent numerical errors that cause beta > pi
            beta = np.linspace(0, np.pi, 2 * b + 1)
            alpha = np.linspace(0, 2 * np.pi, 2 * b + 2, endpoint=False)
        elif grid_type == 'Gauss-Legendre':
            x, _ = leggauss(b + 1)  # TODO: leggauss docs state that this may not be only stable for orders > 100
            beta = np.arccos(x)
            alpha = np.arange(2 * b + 2) * np.pi / (b + 1)
        elif grid_type == 'HEALPix':
            #TODO: implement this here so that we don't need the dependency on healpy / healpix_compat
            from healpix_compat import healpy_sphere_meshgrid
            return healpy_sphere_meshgrid(b)
        elif grid_type == 'equidistribution':
            raise NotImplementedError('Not implemented yet; see Fast evaluation of quadrature formulae on the sphere.')
        else:
            raise ValueError('Unknown grid_type:' + grid_type)
        return beta, alpha


    def rand_rotation_matrix(deflection=1.0, randnums=None):
        """
        Creates a random rotation matrix.
        deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
        rotation. Small deflection => small perturbation.
        randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
        # http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
        """

        if randnums is None:
            randnums = np.random.uniform(size=(3,))

        theta, phi, z = randnums

        theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0*np.pi  # For direction of pole deflection.
        z = z * 2.0*deflection  # For magnitude of pole deflection.

        # Compute a vector V used for distributing points over the sphere
        # via the reflection I - V Transpose(V).  This formulation of V
        # will guarantee that if x[1] and x[2] are uniformly distributed,
        # the reflected points will be uniform on the sphere.  Note that V
        # has length sqrt(2) to eliminate the 2 in the Householder matrix.

        r = np.sqrt(z)
        V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z)
        )

        st = np.sin(theta)
        ct = np.cos(theta)

        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

        # Construct the rotation matrix  ( V Transpose(V) - I ) R.

        M = (np.outer(V, V) - np.eye(3)).dot(R)
        return M


    def rotate_grid(rot, grid):
        x, y, z = grid
        xyz = np.array((x, y, z))
        x_r, y_r, z_r = np.einsum('ij,jab->iab', rot, xyz)
        return x_r, y_r, z_r

    def get_projection_grid(b, grid_type="Driscoll-Healy"):
        ''' returns the spherical grid in euclidean
        coordinates, where the sphere's center is moved
        to (0, 0, 1)'''
        theta, phi = meshgrid(b=b, grid_type=grid_type)
        x_ = np.sin(theta) * np.cos(phi)
        y_ = np.sin(theta) * np.sin(phi)
        z_ = np.cos(theta)
        return x_, y_, z_

    def project_sphere_on_xy_plane(grid, projection_origin):
        ''' returns xy coordinates on the plane
        obtained from projecting each point of
        the spherical grid along the ray from
        the projection origin through the sphere '''

        sx, sy, sz = projection_origin
        x, y, z = grid
        z = z.copy() + 1

        t = -z / (z - sz)
        qx = t * (x - sx) + x
        qy = t * (y - sy) + y

        xmin = 1/2 * (-1 - sx) + -1
        ymin = 1/2 * (-1 - sy) + -1

        # ensure that plane projection
        # ends up on southern hemisphere
        rx = (qx - xmin) / (2 * np.abs(xmin))
        ry = (qy - ymin) / (2 * np.abs(ymin))

        return rx, ry


    def sample_within_bounds(signal, x, y, bounds):
        ''' '''
        xmin, xmax, ymin, ymax = bounds

        idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

        if len(signal.shape) > 2:
            sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]))
            sample[:, idxs] = signal[:, x[idxs], y[idxs]]
        else:
            sample = np.zeros((x.shape[0], x.shape[1]))
            sample[idxs] = signal[x[idxs], y[idxs]]
        return sample


    def sample_bilinear(signal, rx, ry):
        ''' '''

        signal_dim_x = signal.shape[1]
        signal_dim_y = signal.shape[2]

        rx *= signal_dim_x
        ry *= signal_dim_y

        # discretize sample position
        ix = rx.astype(int)
        iy = ry.astype(int)

        # obtain four sample coordinates
        ix0 = ix - 1
        iy0 = iy - 1
        ix1 = ix + 1
        iy1 = iy + 1

        bounds = (0, signal_dim_x, 0, signal_dim_y)

        # sample signal at each four positions
        signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
        signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
        signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
        signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

        # linear interpolation in x-direction
        fx1 = (ix1-rx) * signal_00 + (rx-ix0) * signal_10
        fx2 = (ix1-rx) * signal_01 + (rx-ix0) * signal_11

        # linear interpolation in y-direction
        return (iy1 - ry) * fx1 + (ry - iy0) * fx2


    def project_2d_on_sphere(signal, grid, projection_origin=None):
        if projection_origin is None:
            projection_origin = (0, 0, 2 + NORTHPOLE_EPSILON)
        #project sphere grid which is in Euclidean space to a 2d grid though projection origin which is the noth pole
        rx, ry = project_sphere_on_xy_plane(grid, projection_origin)
        #sample and interpolation in x,y direction
        sample = sample_bilinear(signal, rx, ry)

        # ensure that only south hemisphere gets projected
        sample *= (grid[2] <= 1).astype(np.float64)

        # rescale signal to [0,1]
        sample_min = sample.min(axis=(1, 2)).reshape(-1, 1, 1)
        sample_max = sample.max(axis=(1, 2)).reshape(-1, 1, 1)

        sample = (sample - sample_min) / (sample_max - sample_min)
        sample *= 255
        sample = sample.astype(np.uint8)

        return sample

    def xyz2latlong(vertices):
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        long = np.arctan2(y, x)
        xy2 = x**2 + y**2
        lat = np.arctan2(z, np.sqrt(xy2))
        return lat, long


    def interp_r2tos2(sig_r2, V, method="linear", dtype=np.float32):
        """
        sig_r2: rectangular shape of (lat, long, n_channels)
        V: array of spherical coordinates of shape (n_vertex, 3)
        method: interpolation method. "linear" or "nearest"
        """
        ele, azi = xyz2latlong(V)
        nlat, nlong = sig_r2.shape[0], sig_r2.shape[1]
        dlat, dlong = np.pi/(nlat-1), 2*np.pi/nlong
        lat = np.linspace(-np.pi/2, np.pi/2, nlat)
        long = np.linspace(-np.pi, np.pi, nlong+1)
        sig_r2 = np.concatenate((sig_r2, sig_r2[:, 0:1]), axis=1)
        intp = RegularGridInterpolator((lat, long), sig_r2, method=method)
        s2 = np.array([ele, azi]).T
        sig_s2 = intp(s2).astype(dtype)
        return sig_s2



    trainset = datasets.MNIST(root ='MNIST',train=True, download=True)
    testset = datasets.MNIST(root ='MNIST',train=False, download=True)



    mnist_train = {}
    mnist_train['images'] = trainset.data.numpy()
    mnist_train['labels'] = trainset.targets.numpy()
    mnist_test = {}
    mnist_test['images'] = testset.data.numpy()
    mnist_test['labels'] = testset.targets.numpy()

    grid = get_projection_grid(b=30)
    # below we rotate the grid
    rot = rand_rotation_matrix(deflection=1.0)
    new_grid = rotate_grid(rot, grid)

    dataset = {}
    for label, data in zip(["train", "test"], [mnist_train, mnist_test]):

        print("projecting {0} data set".format(label))
        current = 0
        signals = data['images'].reshape(-1, 28, 28).astype(np.float64)
        n_signals = signals.shape[0]
        projections = np.ndarray(
            (signals.shape[0], 2 * 30, 2 * 30),
            dtype=np.uint8)

        while current < n_signals:
            #-----------------------------------------------------------------------------------------------------
            #below select roatted grid or non-rotated grid
            #rotated_grid = grid is non rotated
            #rotated_grid = new_grid is rotated
            rotated_grid = new_grid
            idxs = np.arange(current, min(n_signals,
                                        current + 500))
            chunk = signals[idxs]
            projections[idxs] = project_2d_on_sphere(chunk, rotated_grid)
            current += 500
            print("\r{0}/{1}".format(current, n_signals), end="")
        print("")
        dataset[label] = {
            'images': projections,
            'labels': data['labels']
        }

    x_train = dataset['train']['images']
    x_test = dataset['test']['images']
    y_train = dataset['train']['labels']
    y_test = dataset['test']['labels']

    # if desired, save raw MNIST to reuse on different point clouds
    np.save(os.path.join(out_path,'MNIST_train_rotated_raw'), x_train)
    np.save(os.path.join(out_path,'MNIST_train_label_raw'), y_train)
    np.save(os.path.join(out_path,'MNIST_test_rotated_raw'), x_test)
    np.save(os.path.join(out_path,'MNIST_test_label_raw'), y_test)

    # generate N uniformly i.i.d. points on unit sphere
    X = np.random.normal(size=(N, 3))
    X = np.divide(X, np.linalg.norm(X, axis=1, keepdims=True))
    x_train_s2 = []
    print("Converting training set...")
    for i in range(x_train.shape[0]):
        x_train_s2.append(interp_r2tos2(x_train[i], X))
    x_test_s2 = []
    print("Converting test set...")
    for i in range(x_test.shape[0]):
        x_test_s2.append(interp_r2tos2(x_test[i], X))
    x_train_s2 = np.stack(x_train_s2, axis=0)
    x_test_s2 = np.stack(x_test_s2, axis=0)

    # save projected dataset
    np.save(os.path.join(out_path,'MNIST_train_rotated_s2'), x_train_s2)
    np.save(os.path.join(out_path,'MNIST_test_rotated_s2'), x_test_s2)
    
    data_list = []
    for i in range(x_train.shape[0]):
        data = Data(x = torch.tensor(x_train_s2[i], dtype=torch.float)[:,None],
                     y = torch.tensor(y_train[i], dtype=torch.long),
                     pos = torch.tensor(X, dtype=torch.float))
        data_list.append(data)
    
    for i in range(x_test.shape[0]):
        data = Data(x = torch.tensor(x_test_s2[i], dtype=torch.float)[:,None],
                     y = torch.tensor(y_test[i], dtype=torch.long),
                     pos = torch.tensor(X, dtype=torch.float))
        data_list.append(data)
    
    return data_list


class InMemoryExt(InMemoryDataset):
    def __init__(
        self,
        root: str,
        transform = None,
        pre_transform = None,
        pre_filter = None,
        normalize_scattering_features = True,
        reprocess_if_different = True,
        njobs = 1,
    ):
        self.njobs = njobs
        self.normalize_scattering_features = normalize_scattering_features
        self.reprocess_if_different = reprocess_if_different
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.pt']
    @property 
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        data_list = create_raw_data(self.raw_dir)
        full_path = os.path.join(self.raw_dir, 'data.pt')
        torch.save(data_list,full_path)

    def process(self):
        data_list = torch.load(os.path.join(self.raw_dir,self.raw_file_names[0]))
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
                print("Normalizing scattering features....")
                if hasattr(data_list[0],"scattering_features"):
                    scat_feats = []
                    d_list = []
                    for d in data_list:
                        scat_feats.append(d.scattering_features)
                    scat_feats = torch.cat(scat_feats)
                    m_scat = scat_feats.mean(0)[None,...]
                    std_scat = scat_feats.std(0)[None,...]
                    max_scat = scat_feats[~torch.isinf(scat_feats)].max()

                    for d in data_list:
                        d.scattering_features = d.scattering_features - m_scat / std_scat
                        d.scattering_features[torch.isinf(d.scattering_features)] = max_scat
                        d_list.append(d) 
                    data_list = d_list

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 
    
    def _process(self):
        f = osp.join(self.processed_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            print(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first")
            if self.reprocess_if_different:
                print("Reprocessing dataset...")
                shutil.rmtree(self.processed_dir)

        f = osp.join(self.processed_dir, 'pre_filter.pt')
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

        path = osp.join(self.processed_dir, 'pre_transform.pt')
        torch.save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.pt')
        torch.save(_repr(self.pre_filter), path)

        if self.log and 'pytest' not in sys.modules:
            print('Done!', file=sys.stderr)

class MNISTData(pl.LightningDataModule):
    def __init__(self, batch_size = 32, num_workers = 4, pin_memory = True, random_state = 42, re_precompute = True, njobs = 1, reprocess_if_different = True, **kwargs):
        """
        k: number of nearest neighbors to consider
        n_samples: number of samples for each point cloud
        """
        super().__init__()
        self.save_hyperparameters()

        dataname = "MNIST"
        def signal_transform(x):
            x.x = x.pos
            return x
        
        if re_precompute:
            if os.path.isdir(os.path.join(DATA_DIR,dataname,"processed")):
                shutil.rmtree(os.path.join(DATA_DIR,dataname,"processed"))
        
        base_pre_transform = [T.NormalizeScale() ]
        pre_transform_list = get_pretransforms(pre_transforms_base = base_pre_transform, fixed_pos = True, **kwargs["graph_construct"])
        pre_transform = T.Compose(pre_transform_list)

        #transform = T.Compose([signal_transform]) # setting the signal as the position of the points
        transform = None
        dataset = InMemoryExt(
            root= os.path.join(DATA_DIR,dataname),
            transform=transform,
            pre_transform=pre_transform,
            njobs = njobs,
            reprocess_if_different = reprocess_if_different
        )

        train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=random_state)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=random_state)

        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset = Subset(dataset, val_idx)
        self.test_dataset = Subset(dataset, test_idx)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.input_dim = 1
        self.num_classes = 10

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
                          collate_fn = laplacian_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,
                          shuffle = False,
                          collate_fn = laplacian_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,
                          shuffle = False,
                          collate_fn = laplacian_collate_fn)


if __name__ == "__main__":
    mdata = MNISTData()
    breakpoint()
    create_raw_data(os.path.join(DATA_DIR,"MNIST"))