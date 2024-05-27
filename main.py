
import os
import time
from utils.loggers import (
    logger, 
    update_log_name,
    update_base_dir, 
    get_base_dir
)

import numpy as np
import torch

import argparse

from algos.spectral_representation import Spec_Repr
from utils.util import load_config

from algos.embedding_sgd_iv import Embedding_SGD

from dataset.demand_design_dataset import DemandDesign_Dataset
from dataset.dsprites_dataset import DSprites_Dataset

import pandas as pd

def build_dataset(config):
    task = config['task']
    if task.lower().startswith('demand_design'):
        return DemandDesign_Dataset(
                    config_dataset=config['dataset'],
                    inference_mode=config['inference_mode'],
                    rho = 0.1,
                    )
    elif task.lower().startswith('dsprites'):
        return DSprites_Dataset(
                    config_dataset=config['dataset'],
                    data_path='dataset/', 
                    inference_mode=config['inference_mode'],
                    )
    else:
        raise NotImplementedError


def experiment(config, round, debug=False):

    logger.info(f'Dataset: {config["task"]}, device: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}')
    logger.info("Config: " + "\n".join(f"{key}: {value}" for key, value in config.items()))

# ===================================
# Build dataset
# ===================================
    dataset = build_dataset(config)
    logger.info(f'Built dataset: {config["task"]}')

# ===================================
# Representation learning + SGD
# ===================================
    if config['algo'] == 'ctrl':
        repr_algo = Spec_Repr(
                    task=config['task'],
                    dims=dataset.get_dims(),
                    config_network=config['network'],
                    inference_mode=config['inference_mode'],
                    )
        
        start_time = time.time()
        repr_algo.train_stage1(dataset, config_network=config['network'])
        loss = repr_algo.train_stage2(dataset, config_sgd=config['sgd'])
    
    elif config['algo'] == 'vanilla':
        repr_algo = Embedding_SGD(dataset=dataset)
        start_time = time.time()
        loss = repr_algo.optimize(config_sgd=config['sgd'])

    else:
        raise NotImplementedError

    run_time = time.time() - start_time
    logger.info(f"--- Run Time: {run_time: 4f} seconds --- ")
    return loss, run_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/dsprites_low_32.yaml")
    args = parser.parse_args()

    config_path = args.config
    # check if the path is valid
    assert os.path.isfile(config_path), f"Invalid config path: {config_path}"

    config = load_config(config_path)

    seed = config["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    results = []
    
    # ===================================
    # Logging / config
    # ===================================
    if config['algo'] == 'ctrl':
        log_base = (
            f'{config["task"]}_{"high" if config["dataset"]["use_high_dim_obs"] else "low"}'
            f'_dim_{config["network"]["phi_dims"][-1]}'
            f'_train_{config["dataset"]["train_size"]}_unlabeled_{config["dataset"]["unlabeled_size"]}'
        )
    elif config['algo'] == 'vanilla':
        log_base = (
            f'{config["task"]}_{"high" if config["dataset"]["use_high_dim_obs"] else "low"}'
            f'_train_{config["dataset"]["train_size"]}'
        )
    else:
        raise NotImplementedError

    update_base_dir(log_base)
    
    for round in range(1):
        update_log_name(round)
        loss, run_time = experiment(config, round, debug=True)
        results.append({'loss': loss, 'run_time': run_time})
        
    results_df = pd.DataFrame(results)
    mean_row = results_df.mean().to_frame().T
    std_row = results_df.std().to_frame().T
    mean_row['description'] = 'mean'
    std_row['description'] = 'std'
    mean_row.reset_index(drop=True, inplace=True)
    std_row.reset_index(drop=True, inplace=True)
    results_df = pd.concat([results_df, mean_row, std_row], ignore_index=True)

    print(results_df)
    results_df.to_csv(f'{get_base_dir()}/results.csv', index=False)
        
        
