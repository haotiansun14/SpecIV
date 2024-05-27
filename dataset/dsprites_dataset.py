'''
This file is used to generate experimental data based on the dSprites dataset.
Original dataset: L. Matthey, I. Higgins, D. Hassabis, and A. Lerchner. dSprites: Disentanglement testing sprites dataset, 2017. 
URL https://github.com/deepmind/dsprites-dataset/.
'''

import numpy as np
from numpy.random import default_rng
import torch
import itertools
from filelock import FileLock

from dataset.base_dataset import Base_Dataset

from utils.util import attach_image, shrink_input

# Adopted from https://github.com/liyuan9988/DeepFeatureIV/tree/master 
def image_id(latent_bases: np.ndarray, posX_id_arr: np.ndarray, posY_id_arr: np.ndarray,
             orientation_id_arr: np.ndarray,
             scale_id_arr: np.ndarray):
    data_size = posX_id_arr.shape[0]
    color_id_arr = np.array([0] * data_size, dtype=int)
    shape_id_arr = np.array([2] * data_size, dtype=int)
    idx = np.c_[color_id_arr, shape_id_arr, scale_id_arr, orientation_id_arr, posX_id_arr, posY_id_arr]
    return idx.dot(latent_bases)

def structural_func(image, weights):
    return (np.mean((image.dot(weights)) ** 2, axis=1) - 5000) / 1000

class DSprites_Dataset(Base_Dataset):
    '''
    Fixed var   color: 1
                shape: 3 (heart)
    Treatment   X:  (B, 4096, 1)
    Outcome     Y:  (B, 1)
    Instrument  Z:  (B, 3)
                Z:  scale --> [0.5, 1]          6 values
                Z:  orientation --> [0, 6.28]   40 values
                Z:  posX --> [0, 1]             32 values 
    Confounder  e:  (B, 1)
                e:  posY --> [0, 1]             32 values    
    inference_mode:
        - IV
        - PCL
    '''
    def __init__(self, 
                 config_dataset,
                 data_path='dataset/', 
                 use_sprite='heart',
                 inference_mode='IV_NO_OBS_CONFOUNDING'
                 ):
        
        test_size = config_dataset['test_size']
        train_size = config_dataset['train_size']
        batch_size = config_dataset['batch_size']
        seed = config_dataset['seed']
        unlabeled_size = config_dataset['unlabeled_size']
        stage1_size = unlabeled_size + train_size

        # random seed
        rng = default_rng(seed)

        with FileLock("./data.lock"):
            dataset_zip = np.load(data_path + "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", 
                                allow_pickle=True, encoding="bytes")
            weights = np.load(data_path + "dsprite_mat.npy")

        imgs = dataset_zip['imgs']
        latents_values = dataset_zip['latents_values']
        metadata = dataset_zip['metadata'][()]

        latents_sizes = metadata[b'latents_sizes']
        latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))

        posX_id_arr = rng.integers(32, size=stage1_size+test_size)
        posY_id_arr = rng.integers(32, size=stage1_size+test_size)
        scale_id_arr = rng.integers(6, size=stage1_size+test_size)
        orientation_arr = rng.integers(40, size=stage1_size+test_size)
        image_idx_arr = image_id(latents_bases, posX_id_arr, posY_id_arr, orientation_arr, scale_id_arr)
        treatment = imgs[image_idx_arr].reshape((stage1_size+test_size, 64 * 64)).astype(np.float64)
        treatment += rng.normal(0.0, 0.1, treatment.shape)
        latent_feature = latents_values[image_idx_arr]  # (color, shape, scale, orientation, posX, posY)
        structural = structural_func(treatment, weights)

        if "iv" in inference_mode.lower():
            instrumental = latent_feature[:, 2:5]  # (scale, orientation, posX)
            outcome_noise = (posY_id_arr - 16.0) + rng.normal(0.0, 0.5, stage1_size+test_size)
            outcome = structural + outcome_noise
            structural = structural[:, np.newaxis]
            outcome = outcome[:, np.newaxis]

            self.X = torch.tensor(treatment, dtype=torch.float32)
            self.Z = torch.tensor(instrumental, dtype=torch.float32)
            self.Y = torch.tensor(outcome, dtype=torch.float32)
            self.f_structure = torch.tensor(structural, dtype=torch.float32)

            # O takes 0 for this dataset as there's no observable confounder
            self.O = torch.zeros_like(self.Y)
            
        elif inference_mode.lower() == 'pcl':
            treatment_proxy = latent_feature[:, 2:5]  # (scale, orientation, posX)
            posX_id_proxy = np.array([16] * (stage1_size+test_size))
            scale_id_proxy = np.array([3] * (stage1_size+test_size))
            orientation_proxy = np.array([0] * (stage1_size+test_size))
            proxy_image_idx_arr = image_id(latents_bases, posX_id_proxy, posY_id_arr, orientation_proxy, scale_id_proxy)
            outcome_proxy = imgs[proxy_image_idx_arr].reshape((stage1_size+test_size, 64 * 64)).astype(np.float64)
            outcome_proxy += rng.normal(0.0, 0.1, outcome_proxy.shape)
            
            outcome = structural * (posY_id_arr - 15.5) ** 2 / 85.25 + rng.normal(0.0, 0.5, stage1_size+test_size)
            outcome = outcome[:, np.newaxis]
            structural = structural[:, np.newaxis]
            # =================================================================
            # To obtain the bridge function h, we adopt the following convention
            # such that we can reuse the spectral_iv framework.
            # Convention:
            # X <- W (outcome proxy), dim: (B, 4096)
            # Z <- Z (treatment proxy), dim: (B, 3)
            # O <- A (treatment), dim: (B, 4096)
            # Y <- outcome as usual, dim: (B, 1)
            # =================================================================
            self.X = torch.tensor(outcome_proxy, dtype=torch.float32)
            self.Z = torch.tensor(treatment_proxy, dtype=torch.float32)
            self.O = torch.tensor(treatment, dtype=torch.float32)
            self.Y = torch.tensor(outcome, dtype=torch.float32)
            self.f_structure = torch.tensor(structural, dtype=torch.float32)
        
        else:
            raise NotImplementedError


        if config_dataset['use_high_dim_obs']:
            # scale: [0.5, 1]
            # orientation: [0, 6.28]
            # posX: [0, 1]
            z_range = [[0.5, 1], [0, 6.28], [0, 1]]
            shrink_func = []
            for r in z_range:
                shrink_func.append(shrink_input(r))
            
            for i in range(3):
                self.Z[:, i] = shrink_func[i](self.Z[:, i])
            
            Z_imgs = torch.zeros(self.Z.shape[0], 28 * 28 * 3)
            for i in range(3):
                # replace with attach image
                Z_imgs[:stage1_size, i * 28 * 28: (i + 1) * 28 * 28] = attach_image(self.Z[:stage1_size, i], train_flg=True, seed=seed)
                Z_imgs[stage1_size:, i * 28 * 28: (i + 1) * 28 * 28] = attach_image(self.Z[stage1_size:, i], train_flg=False, seed=seed)
            self.Z = Z_imgs

            assert self.Z.shape[1] == 28 * 28 * 3
        
        super(DSprites_Dataset, self).__init__(
            seed=seed,
            X=self.X, Z=self.Z, O=self.O, f_structure=self.f_structure, Y=self.Y,
            train_size=train_size, stage1_size=stage1_size, test_size=test_size, batch_size=batch_size
        )


    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def grid_sample_idx(self, latents_sizes, num_samples_per_latent):
        '''
        For each axis of Z, evenly sample num_samples from X and Z
        Note that Z = (B, 4), the second dim corresponds to
        ('scale', 'orientation', 'posX', 'posY')
        the range of each dim is:
        scale --> [0.5, 1]          6 values
        orientation --> [0, 6.28]   40 values
        posX --> [0, 1]             32 values
        posY --> [0, 1]             32 values
        '''
        # get indices for m evenly spaced points in n points
        f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        # get indices for each dimension
        indices = [f(num_samples_per_latent[i], size) for i, size in enumerate(latents_sizes)]

        latent_combs = list(itertools.product(*indices))
        latent_combs = torch.tensor([list(item) for item in latent_combs])

        indices_sampled = self.latent_to_index(latent_combs)
        
        return indices_sampled

