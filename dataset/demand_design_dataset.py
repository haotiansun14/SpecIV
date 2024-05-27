'''
This file is used to generate synthetic data for the demand design task.
'''

import numpy as np
import torch
from numpy.random import default_rng

from utils.util import attach_image, shrink_input

from dataset.base_dataset import Base_Dataset

torch.manual_seed(0)
np.random.seed(0)

def psi(t: np.ndarray) -> np.ndarray:
    return 2 * ((t - 5) ** 4 / 600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)


def f(p: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 100 + (10 + p) * s * psi(t) - 2 * p

class DemandDesign_Dataset(Base_Dataset):
    '''
    Instrumental variable: Z -> C
    Treatment: X -> P
    Observed confounder: O -> T, S
    Outcome: Y -> f(P, T, S) + epsilon

    Approach:
    X = (X, O) = (P, T, S)
    Z = (Z, O) = (C, T, S)
    '''
    def __init__(self,
                 config_dataset,
                 rho=0.1,
                 inference_mode="IV_WITH_OBS_CONFOUNDING"
                 ):
            
        train_size = config_dataset['train_size']
        test_size = config_dataset['test_size']
        batch_size = config_dataset['batch_size']
        seed = config_dataset['seed']

        unlabeled_size = config_dataset['unlabeled_size']
        stage1_size = unlabeled_size + train_size

        use_high_dim_obs = config_dataset['use_high_dim_obs']
        use_high_dim_treatment = config_dataset['use_high_dim_treatment']

        # random seed
        rng = default_rng(seed)

        # train
        emotion = rng.choice(list(range(1, 8)), stage1_size+test_size)
        time = rng.uniform(0, 10, stage1_size+test_size)
        cost = rng.normal(0, 1.0, stage1_size+test_size)
        noise_price = rng.normal(0, 1.0, stage1_size+test_size)
        noise_demand = rho * noise_price + rng.normal(0, np.sqrt(1 - rho ** 2), stage1_size+test_size)
        price = 25 + (cost + 3) * psi(time) + noise_price
        structural: np.ndarray = f(price, time, emotion).astype(float)
        outcome: np.ndarray = (structural + noise_demand).astype(float)

        # if the high-dimensional observable confounder is used, 
        # we need to attach the image to O.
        if use_high_dim_obs:
            S = torch.zeros((stage1_size+test_size, 28 * 28))
            S[:stage1_size, :] = attach_image(emotion[:stage1_size], train_flg=True, seed=seed)
            S[stage1_size:, :] = attach_image(emotion[stage1_size:], train_flg=False, seed=seed)

        # if the high-dimensional treatment is used,
        # we need to attach the image to X.
        if use_high_dim_treatment:
            shrink_func = shrink_input([10,25])
            price = shrink_func(torch.Tensor(price)).numpy()
            P = torch.zeros((stage1_size+test_size, 28 * 28))
            P[:stage1_size, :] = attach_image(price[:stage1_size], train_flg=True, seed=seed)
            P[stage1_size:, :] = attach_image(price[stage1_size:], train_flg=False, seed=seed)

        if inference_mode == "IV_WITH_OBS_CONFOUNDING":
            # X = P, Z = C, O = (T, S)
            treatment = price
            instrumental = cost
            obs = np.c_[time, emotion]
            self.Z = torch.from_numpy(instrumental).float().reshape(-1, 1)
            self.O = torch.from_numpy(obs).float()

            if use_high_dim_obs:
                self.O = torch.cat([self.O[:, :1], S], dim=1)
            
            if use_high_dim_treatment:
                self.X = P
            else:
                self.X = torch.from_numpy(treatment).float().reshape(-1, 1)

        else:
            treatment = np.c_[price, time, emotion]
            instrumental = np.c_[cost, time, emotion]

            # X = (X, O) = (P, T, S)
            if use_high_dim_treatment:
                self.X = P
            else:
                self.X = torch.from_numpy(treatment).float()
            # Z = (Z, O) = (C, T, S)
            self.Z = torch.from_numpy(instrumental).float()
            self.O = torch.zeros_like(self.X)
            T = torch.from_numpy(time).float().reshape(-1, 1)
            if use_high_dim_obs:
                self.X = torch.cat([self.X[:, :28 * 28], T, S], dim=1)
                self.Z = torch.cat([self.Z[:, :28 * 28 + 1], S], dim=1)

        # Y = (Y, O) = (f(P, T, S) + epsilon, T, S)
        self.Y = torch.from_numpy(outcome[:, np.newaxis]).float()
        # f_structure = (f(P, T, S), T, S)
        self.f_structure = torch.from_numpy(structural[:, np.newaxis]).float()

        super(DemandDesign_Dataset, self).__init__(
            seed=seed,
            X=self.X, Z=self.Z, O=self.O, f_structure=self.f_structure, Y=self.Y,
            train_size=train_size, stage1_size=stage1_size, test_size=test_size, batch_size=batch_size
        )



    




