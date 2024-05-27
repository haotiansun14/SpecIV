import numpy as np
import torch
from torch import nn, bmm
import torch.nn.functional as F

from datetime import datetime
from utils.loggers import logger

from tqdm.auto import tqdm

from networks.contrastive_models import Phi, Mu, Xi, Nu
from utils.lin_alg import k_prod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Using contrastive learning (CTRL) to solve the IV/PCL problem.
'''
class Spec_Repr():
# ===================================
# Initialization for three modes:
#     - IV_NO_OBS_CONFOUNDING
#     - IV_WITH_OBS_CONFOUNDING
#     - PCL
# ===================================

    def __init__(
            self, 
            task,
            dims, 
            config_network,
            inference_mode="IV_NO_OBS_CONFOUNDING",
            ):
        # seed = config_network.get('seed', 0)
        # torch.manual_seed(seed)
        
        lr = config_network['lr']

        self.gamma = config_network['gamma']

        phi_dims = config_network['phi_dims']
        mu_dims = config_network['mu_dims']
        nu_dims = config_network.get('nu_dims', None)
        xi_dims = config_network.get('xi_dims', None)
  
        self.use_image_feature = config_network['use_image_feature']

        self.x_feat_dim = phi_dims[-1]
        self.inference_mode = inference_mode
        

        if self.inference_mode=="IV_WITH_OBS_CONFOUNDING":
            self.o_feat_dim = xi_dims[-1]
            self.z_feat_dim = mu_dims[-1]
            self.y_feat_dim = nu_dims[-1]
   
            # We have four representations, i.e.,
            # phi(x), mu(z), xi(o), nu(y)
            self.phi = Phi(
                x_dim=dims['x_dim'], 
                network_dims=phi_dims,
                use_image_feature=True,
                ).to(device)

            self.mu = Mu(
                z_dim=dims['z_dim'],
                network_dims=mu_dims,
                use_image_feature=False
                ).to(device)

            self.nu_aux = Nu(
                y_dim=1,
                network_dims=nu_dims,
                use_image_feature=False
                ).to(device)

            # universal parametrization for observables
            self.xi = Xi(
                o_dim=dims['o_dim'],
                network_dims=xi_dims,
                use_image_feature=self.use_image_feature
                ).to(device)

            # ================================================================
            # RELATIONS
            # ----------------------------------------------------------------
            # We have three relations
            # P(x|o,z) = <phi(x), V(o) @ mu(z)>
            #                  [d_x]  [d_x * d_z]  [d_z]
            # P(y|o,z) = <nu(y), W(o) @ mu(z)>
            #                  [d_y]  [d_y * d_z]  [d_z]
   
            # We also have two tunable variables for the simplification, i.e.,
            # V(o) = P_V \otimes_3 xi(o)                 | P_V: d_x * d_z * d_o
            # [d_x * d_z]    [d_x * d_z * d_o] [d_o]
            # W(o) = Q(o).T @ V(o)                    
            # [d_y * d_z]    [d_y*d_x]  [d_x*d_z]         
            # Q(o) = P_Q \otimes_3 xi(o)                | P_Q: d_x * d_y * d_o
            # [d_x * d_y]    [d_x * d_y * d_o] [d_o]
            # ================================================================
   
            self.P_V = torch.randn(
                    self.x_feat_dim, self.z_feat_dim, self.o_feat_dim,
                    requires_grad=True, 
                    device=device
                  )
            self.P_Q = torch.randn(
                    self.x_feat_dim, self.y_feat_dim, self.o_feat_dim,
                    requires_grad=True, 
                    device=device
                )

            v_dim = self.x_feat_dim * self.o_feat_dim
            w_dim = self.z_feat_dim * self.o_feat_dim * self.o_feat_dim
            self.v = torch.zeros(v_dim, 1, requires_grad=True, device=device)
            self.w = torch.zeros(w_dim, 1, requires_grad=True, device=device)

            self.feature_optimizer = torch.optim.Adam(
                list(self.phi.parameters()) + list(self.mu.parameters()) +  \
                list(self.nu_aux.parameters()) + list(self.xi.parameters()) + \
                  [self.P_V] + [self.P_Q],
                weight_decay=1e-4, lr=lr)
   
            print(f"Feature dim: {self.x_feat_dim}, o_feat_dim: {self.o_feat_dim}, z_feat_dim: {self.z_feat_dim}, y_feat_dim: {self.y_feat_dim}")
            print(f"P_V: {self.P_V.shape}, P_Q: {self.P_Q.shape}, v: {self.v.shape}, w: {self.w.shape}")
            

        elif self.inference_mode=="IV_NO_OBS_CONFOUNDING":

            self.phi = Phi(x_dim=dims['x_dim'], 
                    network_dims=phi_dims,
                    use_image_feature=self.use_image_feature
                    ).to(device)

            self.mu = Mu(z_dim=dims['z_dim'],
                    network_dims=mu_dims,
                    use_image_feature=self.use_image_feature
                    ).to(device)

            self.v = torch.zeros(self.x_feat_dim, 1, requires_grad=True, device=device)
            self.w = torch.zeros(self.x_feat_dim, 1, requires_grad=True, device=device)

            self.feature_optimizer = torch.optim.Adam(
                list(self.phi.parameters()) + list(self.mu.parameters()),
                weight_decay=1e-4, lr=lr)
            
        
        elif self.inference_mode=="PCL":
            self.o_feat_dim = xi_dims[-1]
            self.z_feat_dim = mu_dims[-1]
            self.y_feat_dim = nu_dims[-1]
            
            # parametrization of P(x|z,o)
            self.phi = Phi(x_dim=dims['x_dim'], 
                    network_dims=phi_dims,
                    use_image_feature=self.use_image_feature
                    ).to(device)

            self.mu = Mu(z_dim=dims['z_dim'],
                    network_dims=mu_dims,
                    use_image_feature=self.use_image_feature
                    ).to(device)
            
            self.nu_aux = Nu(
                    y_dim=1,
                    network_dims=nu_dims,
                    use_image_feature=False,
                    ).to(device)

            # universal parametrization for both P(x|z,o) and P(y|x,o)
            self.xi = Xi(
                    o_dim=dims['o_dim'],
                    network_dims=xi_dims,
                    use_image_feature=self.use_image_feature,
                    ).to(device)
       
            self.P_V = torch.randn(
                    self.x_feat_dim, self.z_feat_dim, self.o_feat_dim,
                    requires_grad=True, 
                    device=device
                  )
            self.P_Q = torch.randn(
                    self.x_feat_dim, self.y_feat_dim, self.o_feat_dim,
                    requires_grad=True, 
                    device=device
                )

            v_dim = self.x_feat_dim * self.o_feat_dim
            w_dim = self.z_feat_dim * self.o_feat_dim * self.o_feat_dim
            self.v = torch.zeros(v_dim, 1, requires_grad=True, device=device)
            self.w = torch.zeros(w_dim, 1, requires_grad=True, device=device)

            self.feature_optimizer = torch.optim.Adam(
                list(self.phi.parameters()) + list(self.mu.parameters()) +  \
                list(self.nu_aux.parameters()) + list(self.xi.parameters()) + \
                  [self.P_V] + [self.P_Q],
                weight_decay=1e-4, lr=lr)
   
            print(f"Feature dim: {self.x_feat_dim}, o_feat_dim: {self.o_feat_dim}, z_feat_dim: {self.z_feat_dim}, y_feat_dim: {self.y_feat_dim}")
            print(f"P_V: {self.P_V.shape}, P_Q: {self.P_Q.shape}, v: {self.v.shape}, w: {self.w.shape}")

        else:
            raise NotImplementedError(f"Unknown inference mode: {self.inference_mode}")

# ===================================
# Getters for three modes:
# ===================================
    def get_f(self, x, o=None):
        if self.inference_mode in ["IV_WITH_OBS_CONFOUNDING", "PCL"]:
            phi_x = self.phi(x)
            xi_o = self.xi(o)
            phi_xi = k_prod(phi_x, xi_o, mode='all')
            return phi_xi @ self.v
        
        if self.inference_mode == "IV_NO_OBS_CONFOUNDING":
            phi_x = self.phi(x)
            return phi_x @ self.v
        
    def get_exp_f(self, z, o=None):
        if self.inference_mode in ["IV_WITH_OBS_CONFOUNDING", "PCL"]:
            mu_z = self.mu(z)
            xi_o = self.xi(o)
            mu_xi_xi = k_prod(mu_z, xi_o, mode='all')
            mu_xi_xi = k_prod(mu_xi_xi, xi_o, mode='all')
            return mu_xi_xi @ self.w
        
        if self.inference_mode=="IV_NO_OBS_CONFOUNDING":
            mu_z = self.mu(z)
            return mu_z @ self.w
    
# ===================================
# Stage 1: train to find spectral repr
# ===================================
# Convention: P(repr0 | repr1) = <Feat_map0(repr0), Feat_map1(repr1)>
    def contrastive_loss(self, repr0, repr1):
        assert repr0.shape == repr1.shape, f"repr0 shape: {repr0.shape}, repr1 shape: {repr1.shape}"
        assert repr0.dim() == 2, f"repr0 dim: {repr0.dim()}"
        
        labels = torch.eye(repr0.shape[0]).to(device)
        contrastive = (repr1[:, None, :] * repr0[None, :, :]).sum(-1) / self.gamma
        model_loss = nn.CrossEntropyLoss()
        model_loss = model_loss(contrastive, labels)
        return model_loss
    
    def probability_loss(self, repr0, repr1):
        # probability density constraint
        prob_loss = (repr1 * repr0).sum(-1).mean().clamp(min=1e-6)
        prob_loss = prob_loss.log().square()
        return prob_loss
    
    def max_w(self, x, z, o, y, lambda_w):
        f_rec = self.get_f(x=x, o=o)
        f_exp = self.get_exp_f(z=z, o=o)

        w_loss = f_exp * (y - f_rec) - 0.5 * f_exp.square() 
        w_loss = - w_loss + lambda_w * torch.norm(self.w).square()
        assert w_loss.shape == (x.shape[0], 1)
        w_loss = w_loss.mean()
        
        self.w_optimizer.zero_grad()
        w_loss.backward()
        self.w_optimizer.step()

        return w_loss
    
    def min_v(self, x, z, o, y, lambda_v):
        f_rec = self.get_f(x=x, o=o)
        f_exp = self.get_exp_f(z=z, o=o)
  
        v_loss = f_exp * (y - f_rec) + lambda_v * torch.norm(self.v).square()
        assert v_loss.shape == (x.shape[0], 1)
        v_loss = v_loss.mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        return v_loss
    
    def min_v_closed(self, x, z, o, y, lambda_v):
        f_rec = self.get_f(x=x, o=o)

        v_loss = - y * f_rec + 0.5 * f_rec.square() + lambda_v * torch.norm(self.v).square()
        assert v_loss.shape == (x.shape[0], 1)
        v_loss = v_loss.mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        return v_loss
    
    def feature_step(self, batch_train, batch_test, prob_loss_weight, aux_loss_weight):
        """
        Loss implementation 
        """
        x, z, o, _, y = [b.to(device) for b in batch_train]
        x_test, z_test, o_test, _, _ = [b.to(device) for b in batch_test]
        
        # To compute P(x|z,o)
        phi_x = self.phi(x)
        mu_z = self.mu(z)
        phi_x_test = self.phi(x_test)
        mu_z_test = self.mu(z_test)
        

        if self.inference_mode in ["IV_WITH_OBS_CONFOUNDING", "PCL"]:
            # ================================================================
            # RECAP: RELATIONS
            # ----------------------------------------------------------------
            # We have three relations
            # P(x|o,z) = <phi(x), V(o) @ mu(z)>
            #                  [d_x]  [d_x * d_z]  [d_z]
            # P(y|o,z) = <nu(y), W(o) @ mu(z)>
            #                  [d_y]  [d_y * d_z]  [d_z]
   
            # We also have two tunable variables for the simplification, i.e.,
            # V(o) = P_V \otimes_3 xi(o)                 | P_V: d_x * d_z * d_o
            # [d_x * d_z]    [d_x * d_z * d_o] [d_o]
            # W(o) = Q(o).T @ V(o)                    
            # [d_y * d_z]    [d_y*d_x]  [d_x*d_z]         
            # Q(o) = P_Q \otimes_3 xi(o)                | P_Q: d_x * d_y * d_o
            # [d_x * d_y]    [d_x * d_y * d_o] [d_o]
            # ================================================================
   
            xi_o = self.xi(o)
            xi_o_test = self.xi(o_test)
   
            V_o = k_prod(self.P_V, xi_o, mode='last')
            V_o_test = k_prod(self.P_V, xi_o_test, mode='last')
   
            # To compute P(x|o,z)
            V_mu = V_o.bmm(mu_z[:, :, None]).squeeze(-1)
            V_mu_test = V_o_test.bmm(mu_z_test[:, :, None]).squeeze(-1)
      
            loss_xoz = self.contrastive_loss(phi_x, V_mu)
            loss_xoz_test = self.contrastive_loss(phi_x_test, V_mu_test)
   
            if prob_loss_weight > 0:
                prob_loss = self.probability_loss(phi_x, V_mu)
            else:
                prob_loss = torch.tensor(0.0).to(device)
            
            ml_loss = loss_xoz + prob_loss_weight * prob_loss

            # Skip nan values in y
            if not torch.isnan(y).all():
                valid_idx = ~torch.isnan(y).squeeze()
                valid_y = y[valid_idx, :]
                valid_xi_o = xi_o[valid_idx, :]
                valid_mu_z = mu_z[valid_idx, :]
                nu_y_aux = self.nu_aux(valid_y)

                # To compute P(y|o,z)
                V_o_aux = k_prod(self.P_V, valid_xi_o, mode='last')
                Q_o_aux = k_prod(self.P_Q, valid_xi_o, mode='last')
                
                W_o_aux = Q_o_aux.mT.bmm(V_o_aux)
                W_mu_aux = W_o_aux.bmm(valid_mu_z[:, :, None]).squeeze(-1)

                if prob_loss_weight > 0:
                    prob_loss_aux = self.probability_loss(nu_y_aux, W_mu_aux)
                else:
                    prob_loss_aux = torch.tensor(0.0).to(device)

                loss_aux = self.contrastive_loss(nu_y_aux, W_mu_aux) + prob_loss_weight * prob_loss_aux

            else:
                loss_aux = torch.tensor(0.0).to(device)
   
            loss = ml_loss + aux_loss_weight * loss_aux
      
            # for test loss
            test_loss = loss_xoz_test

            P_V_norm = torch.norm(self.P_V).item()
            P_Q_norm = torch.norm(self.P_Q).item()
            
            assert V_o.shape == (x.shape[0], self.x_feat_dim, self.z_feat_dim), f"V_o shape: {V_o.shape}"
            assert W_o_aux.shape == (valid_y.shape[0], self.y_feat_dim, self.z_feat_dim), f"W_o_aux shape: {W_o_aux.shape}"
            assert Q_o_aux.shape == (valid_y.shape[0], self.x_feat_dim, self.y_feat_dim), f"Q_o_aux shape: {Q_o_aux.shape}"

        elif self.inference_mode == "IV_NO_OBS_CONFOUNDING":
            phi_x = self.phi(x)
            mu_z = self.mu(z)
            
            if prob_loss_weight > 0:
                prob_loss = self.probability_loss(phi_x, mu_z)
            else:
                prob_loss = torch.tensor(0.0).to(device)

            loss = self.contrastive_loss(phi_x, mu_z) + prob_loss_weight * prob_loss
            test_loss = self.contrastive_loss(phi_x_test, mu_z_test)


        self.feature_optimizer.zero_grad()
        loss.backward()
        self.feature_optimizer.step()

        return {
            'train_feature_loss': loss.item(),
            'test_feature_loss': test_loss.item(),
            'prob_loss': prob_loss.item(),
            'main_loss': loss_xoz.item() if 'loss_xoz' in locals() else 0.,
            'aux_loss': loss_aux.item() if 'loss_aux' in locals() else 0., 
            'p1_norm': P_V_norm if 'P_V_norm' in locals() else 0.,
            'p3_norm': P_Q_norm if 'P_Q_norm' in locals() else 0.,
        }

# ===================================
# Stage 2: train at embedding SGD
# ===================================
    def sgd_step(self, batch_train, batch_test, reg_weight, close_form_w=False):
        """
        Loss implementation 
        """
        x, z, o, _, y = [b.to(device) for b in batch_train]
        x_test, _, o_test, f_test, _ = [b.to(device) for b in batch_test]
        
        if close_form_w:
            v_loss = self.min_v_closed(x, z, o, y, reg_weight['lambda_v'])
            w_loss = torch.tensor(0.0).to(device)
        else:
            w_loss = self.max_w(x, z, o, y, reg_weight['lambda_w'])
            v_loss = self.min_v(x, z, o, y, reg_weight['lambda_v'])
        
        # record MSE on train/val/test set
        train_mse_loss = F.mse_loss(self.get_f(x, o), y)
        test_mse_loss = F.mse_loss(self.get_f(x_test, o_test), f_test)
  
        return {
            'test_mse_loss': test_mse_loss.item(),
            'train_mse_loss': train_mse_loss.item(),
            'w_loss': w_loss.item(),
            'v_loss': v_loss.item()
        }
    
# ===================================
# (Extra) PCL loss
# ===================================
    def pcl_step(self, batch_train, batch_val, batch_test, batch_all):
        x_all, _, _, _, _ = [b.to(device) for b in batch_all]
        _, _, o_train, _, y_train = [b.to(device) for b in batch_train]
        _, _, o_val, _, y_val = [b.to(device) for b in batch_val]
        _, _, o_test, f_test, _ = [b.to(device) for b in batch_test]


        phi_x_mean = self.phi(x_all).mean(0).reshape(1, -1)
        phi_x_mean_train = phi_x_mean.tile((o_train.shape[0], 1))
        phi_x_mean_val = phi_x_mean.tile((o_val.shape[0], 1))
        phi_x_mean_test = phi_x_mean.tile((o_test.shape[0], 1))

        f_rec_train = k_prod(phi_x_mean_train, self.xi(o_train), mode="all") @ self.v
        train_pcl_loss = F.mse_loss(f_rec_train, y_train)

        f_rec_val = k_prod(phi_x_mean_val, self.xi(o_val), mode="all") @ self.v
        val_pcl_loss = F.mse_loss(f_rec_val, y_val)
    
        f_rec_test = k_prod(phi_x_mean_test, self.xi(o_test), mode="all") @ self.v
        test_pcl_loss = F.mse_loss(f_rec_test, f_test)

        return {
            'test_pcl_loss': test_pcl_loss.item(),
            'val_pcl_loss': val_pcl_loss.item(),
            'train_pcl_loss': train_pcl_loss.item(),
        }

# ===================================
# Composed training stages
# ===================================
    def train_stage1(self, dataset, config_network, verbose=True):
        """
        train to find spectral representation
        """
        num_iter = config_network['num_iter']
        eval_freq = config_network['eval_freq']
        prob_loss_weight = config_network['prob_loss_weight']
        aux_loss_weight = config_network.get('aux_loss_weight', 0.)

        train_loader, _ = dataset.get_train_loader(mode='stage1')
        batch_test = dataset.get_samples('test')

        for t in tqdm(range(int(num_iter)), desc="Stage 1"):
            for batch_train in train_loader:
                info = self.feature_step(
                    batch_train=batch_train, 
                    batch_test=batch_test, 
                    prob_loss_weight=prob_loss_weight,
                    aux_loss_weight=aux_loss_weight
                 )
    
        



    def train_stage2(self, dataset, config_sgd, verbose=True):
        """
        train at embedding SGD
        """
        num_iter = config_sgd['num_iter']
        eval_freq = config_sgd['eval_freq']
        eta = config_sgd['eta']
        reg_weight = config_sgd['reg_weight']
        dataset.batch_size = config_sgd['batch_size']

        self.v_optimizer = torch.optim.SGD(
            [self.v],
            weight_decay=0, lr=eta)
        
        self.w_optimizer = torch.optim.SGD(
            [self.w],
            weight_decay=0, lr=eta)
        
        if self.inference_mode.lower() == "pcl":
            train_loader, val_loader = dataset.get_train_loader(mode='stage2', use_validation=True)
            batch_val = next(iter(val_loader))
            best_val_pcl_loss = np.inf
            best_test_pcl_loss = np.inf
        else:
            train_loader, _ = dataset.get_train_loader(mode='stage2')

        batch_test = dataset.get_samples('test')
        for t in tqdm(range(int(num_iter)), desc="Stage 2"):
            for batch_train in train_loader:
                info = self.sgd_step(
                        batch_train=batch_train, 
                        batch_test=batch_test,
                        reg_weight=reg_weight,  
                        close_form_w=False)

            if self.inference_mode.lower() == "pcl":
                batch_all = dataset.get_samples('train_all')
                info_pcl = self.pcl_step(batch_train=batch_train, 
                                         batch_val=batch_val, 
                                        batch_test=batch_test, 
                                        batch_all=batch_all)
                info_pcl.update(info)
                info = info_pcl
                # store test and val pcl loss such that we can pick up the best result
                if info['val_pcl_loss'] < best_val_pcl_loss:
                    best_val_pcl_loss = info['val_pcl_loss']
                    best_test_pcl_loss = info['test_pcl_loss']
                # early stopping
                if info['val_pcl_loss'] - best_val_pcl_loss > 0.01:
                    break

            # print(f"Epoch {t+1}" + " | ".join(f"{key}: {value:.4f}" for key, value in info.items()))
            
        return info["test_mse_loss"] if self.inference_mode.lower() != "pcl" else best_test_pcl_loss
 
