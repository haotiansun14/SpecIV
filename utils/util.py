import numpy as np 
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from numpy.random import default_rng
import pathlib
from itertools import product
from torchvision.datasets import MNIST
from torchvision import transforms
from itertools import product

import matplotlib.pyplot as plt


def unpack_batch(batch):
  return batch.state, batch.action, batch.next_state, batch.reward, batch.done


def eval_policy(policy, eval_env, eval_episodes=10):
    """
    Eval a policy
    """
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward



def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                input_dim,
                hidden_dim,
                output_dim,
                hidden_depth,
                output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                                            output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def plot_loss(config_sgd, kres_emb):
    num_iter = int(config_sgd['num_iter_outer'])
    eval_interval = int(config_sgd['eval_freq_outer'])
    batch_size = config_sgd['batch_size']
    x_scale, y_scale = config_sgd['plot']['x_scale'], config_sgd['plot']['y_scale']
    
    # plotting
    eval_timestamps = (np.arange(num_iter // eval_interval + 1) * eval_interval) * batch_size 
    # get the minimum mse between mse and mse_avg
    res_mse_test = np.minimum(kres_emb['mse_test'], kres_emb['mse_test_avg'])
    res_mse_train = np.minimum(kres_emb['mse_train'], kres_emb['mse_train_avg'])

    plt.figure()
    plt.plot(eval_timestamps / x_scale, res_mse_test / y_scale, 'r', linewidth=3)
    plt.plot(eval_timestamps / x_scale, res_mse_train / y_scale, 'b', linewidth=3)
    plt.legend(['Test', 'Train'])
    if x_scale == 1:
        plt.xlabel('Iteration')
    else:
        plt.xlabel(f'Iteration ($\\times 10^{int(np.log10(x_scale))}$)')
    if y_scale == 1:
        plt.ylabel('MSE')
    else:
        plt.ylabel(f'MSE ($\\times 10^{int(np.log10(y_scale))}$)')
    plt.title('MSE vs. Iteration')

    res_mae = np.minimum(kres_emb['mae_test'], kres_emb['mae_test_avg'])
    res_mae_train = np.minimum(kres_emb['mae_train'], kres_emb['mae_train_avg'])

    plt.figure()
    plt.plot(eval_timestamps / x_scale, res_mae, 'r', linewidth=3)
    plt.plot(eval_timestamps / x_scale, res_mae_train, 'b', linewidth=3)
    plt.legend(['Test', 'Train'])
    if x_scale == 1:
        plt.xlabel('Iteration')
    else:
        plt.xlabel(f'Iteration ($\\times 10^{int(np.log10(x_scale))}$)')
    plt.ylabel('MAE')
    plt.title('MAE vs. Iteration')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# Helper function to show images
def show_images_grid(input, num_images=25):
    X, Z, F, Y = input
    imgs_ = X.reshape(-1, 64, 64)
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            # only show two digits
            ax.set_title(f'Z: {Z[ax_i, 0]:.2f}, {Z[ax_i, 1]:.2f}, {Z[ax_i, 2]:.2f}\ne: {F[ax_i, 0]:.2f}|Y: {Y[ax_i, 0]:.2f}')
        else:
            ax.axis('off')

def show_density(X):
    imgs = X.reshape(-1, 64, 64).numpy()
    _, ax = plt.subplots()
    ax.imshow(imgs.mean(axis=0), interpolation='nearest', cmap='Greys_r')
    ax.grid('off')
    ax.set_xticks([])
    ax.set_yticks([])

# modified based on https://github.com/liyuan9988/DeepFeatureIV/
def attach_image(num_array, train_flg, seed=42):
    """
    Randomly samples number from MNIST datasets

    Parameters
    ----------
    num_array : Array[int]
        Array of numbers that we sample for
    train_flg : bool
        Whether to sample from train/test in MNIST dataset
    seed : int
        Random seed
    Returns
    -------
    result : (len(num_array), 28*28) array
        ndarray for sampled images
    """

    # random seed
    rng = default_rng(seed)
    
    mnist = MNIST("./dataset/", train=train_flg, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]), target_transform=None, download=True)
    img_data = mnist.data.numpy()
    img_target = mnist.targets.numpy()

    def select_one(idx):
        idx = int(idx.item())
        sub = img_data[img_target == idx]
        return sub[rng.choice(sub.shape[0])].reshape((1, -1))

    return torch.Tensor(np.concatenate([select_one(num) for num in num_array], axis=0))

def shrink_input(input_range=[0,1]):
    a, b = input_range[0], input_range[1]
    coef = 9 / (b - a)
    b = np.ceil(coef * a)

    return lambda x: torch.minimum(torch.maximum(coef * x - b, torch.zeros_like(x)), 9 * torch.ones_like(x)).int()
