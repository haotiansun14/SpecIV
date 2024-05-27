'''
This file implements the embedding SGD algorithm proposed in
    Bo Dai, Niao He, Yunpeng Pan, Byron Boots, and Le Song. Learning from conditional distributions 
    via dual embeddings. In Artificial Intelligence and Statistics, pages 1458-1467. PMLR, 2017.
'''
import numpy as np
import torch
from utils.kernels import construct_kernel, get_median

from utils.loggers import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Embedding_SGD():
    def __init__(self, 
              dataset
              ):
        '''
        param dataset: dataset object

        '''
        self.dataset = dataset
        self.train_size = dataset.get_train_size()

        self.results = {'mae_train': [], 'mae_test': [], 'mse_train': [], 'mse_test': [],
                          'mae_train_avg': [], 'mae_test_avg': [], 'mse_train_avg': [], 'mse_test_avg': []}

    def optimize(self, 
                config_sgd,
                verbose=True
                ):

        num_iter = int(config_sgd['num_iter'])
        eval_freq = int(config_sgd['eval_freq'])
        num_init = int(config_sgd['num_init'])

        batch_size = config_sgd['batch_size']
        eta = config_sgd['eta']
        p_order = config_sgd['p_order']
        reg_weight = config_sgd['reg_weight']


        x_train, z_train, _, _, y_train = [d.to(device) for d in self.dataset.get_samples('train_all')]
        x_test, _, _, f_s_test, _ = [d.to(device) for d in self.dataset.get_samples('test')]

        weights = 0.0

        repr_primal, repr_dual, repr_test = self.get_repr_matrices(x_train, z_train, x_test)

        v, v_avg, w = self.initialize_repr_vectors()

        logger.info("Start training...")

        for i in range(num_iter):

            v_grad, w_grad = self.get_repr_gradients(repr_primal, repr_dual, y_train, v, w, batch_size)

            # As proposed in (Nemirovski  et al., 2009).
            step_size = eta / (num_init + np.power(i, p_order))

            # Embedding SGD update
            v = (1 - reg_weight['lambda_v'] * step_size) * v - step_size * v_grad
            w = (1 - reg_weight['lambda_w'] * step_size) * w + step_size * w_grad

            # weighted average
            v_avg = v_avg * weights + step_size * v
            weights += step_size
            v_avg /= weights 

            # check convergence
            if i == 0 or (i + 1) % eval_freq == 0:
                f_rec_test = repr_test @ v
                f_rec_avg_test = repr_test @ v_avg
                f_rec_train = repr_primal @ v
                f_rec_avg_train = repr_primal @ v_avg
                self.save_results(f_rec_test, f_rec_avg_test, f_s_test, f_rec_train, f_rec_avg_train, y_train)                

                if verbose:
                    logger.info(f'Iter {i+1}: mse_test {self.results["mse_test"][-1]} | mse_test_avg {self.results["mse_test_avg"][-1]} | mse_train {self.results["mse_train"][-1]} | mse_avg_train {self.results["mse_train_avg"][-1]}')     
        return self.results['mse_test']

    def initialize_repr_vectors(self):
        # By definition, f(x) = <Phi(x), v>  and u(z) = <Mu(z), w>
        v = torch.zeros(self.train_size, 1).to(device)
        v_avg = torch.zeros(self.train_size, 1).to(device)
        w = torch.zeros(self.train_size, 1).to(device)
        return v, v_avg, w

    def get_repr_matrices(self, x_train, z_train, x_test):
        # if Gaussian, kernel_params is {'kernel_type': 'Gaussian', 't': std}
        # use median trick to get the bandwidth

        sigmaX = get_median(x_train.detach().cpu().numpy())
        sigmaZ = get_median(z_train.detach().cpu().numpy())
        primal_params = {'kernel_type': 'Gaussian', 't': sigmaX}
        dual_params = {'kernel_type': 'Gaussian', 't': sigmaZ}

        # Pre-compute the kernel 
        # Primal kernel - f(x) = <v, phi(x)>
        repr_primal = construct_kernel(x_train, None, primal_params).to(device)
        # Dual kernel - u(z) = <w, mu(z)>
        repr_dual = construct_kernel(z_train, None, dual_params).to(device)
        # Primal kernel for testing
        repr_test = construct_kernel(x_test, x_train, primal_params).to(device)

        return repr_primal, repr_dual, repr_test
    
    def get_repr_gradients(self, repr_primal, repr_dual, y_train, v, w, batch_size):
        sample_idx = torch.randint(0, self.train_size, (batch_size,))
        f_temp = repr_primal[sample_idx, :] @ v
        u_temp = repr_dual[sample_idx, :] @ w
        v_grad = torch.zeros(self.train_size, 1).to(device)
        w_grad = torch.zeros(self.train_size, 1).to(device)
        # Calculate the gradient for v and w
        v_grad[sample_idx, :] = - u_temp #- y_train[sample_idx, :] + u_temp 
        w_grad[sample_idx, :] = y_train[sample_idx, :] - f_temp - u_temp #f_temp - u_temp
        return v_grad, w_grad
                
    def save_results(self, f_rec_test, f_rec_avg_test, f_s_test, f_rec_train, f_rec_avg_train, y_train):
        self.results['mse_test'].append((f_rec_test - f_s_test).square().mean().detach().cpu().numpy())
        self.results['mse_test_avg'].append((f_rec_avg_test - f_s_test).square().mean().detach().cpu().numpy())
        self.results['mae_test'].append((f_rec_test - f_s_test).abs().mean().detach().cpu().numpy())
        self.results['mae_test_avg'].append((f_rec_avg_test - f_s_test).abs().mean().detach().cpu().numpy())
        
        self.results['mse_train'].append((f_rec_train - y_train).square().mean().detach().cpu().numpy())
        self.results['mse_train_avg'].append((f_rec_avg_train - y_train).square().mean().detach().cpu().numpy())
        self.results['mae_train'].append((f_rec_train - y_train).abs().mean().detach().cpu().numpy())
        self.results['mae_train_avg'].append((f_rec_avg_train - y_train).abs().mean().detach().cpu().numpy())




            


