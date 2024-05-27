import torch
from torch import nn
from torch.nn import functional as F

from networks.image_models import Image_Feature 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for contrastive loss-based model
class Mu(nn.Module):
    """
    Mu: z -> mu(z) in R^d
    """
    def __init__(
        self, 
        z_dim,
        network_dims=[256, 256, 256],
        use_image_feature=False
        ):

        super(Mu, self).__init__()

        if use_image_feature:
            self.image_feature = Image_Feature(num_dense_feature=2).to(device)
            input_dim = self.image_feature.num_dense_feature + self.image_feature.linear.out_features
        else:
            input_dim = z_dim

        network_dims = [input_dim] + network_dims

        # create network list using network_dims
        self.network = nn.ModuleList()
        self.bn = nn.ModuleList()

        for i in range(len(network_dims)-1):
            self.network.append(nn.Linear(network_dims[i], network_dims[i+1]))
            self.bn.append(nn.BatchNorm1d(network_dims[i+1]))
    
        self.use_image_feature = use_image_feature

    def forward(self, z):
        z = self.image_feature(z) if self.use_image_feature else z
        for i in range(len(self.network)-1):
            z = F.relu(self.network[i](z)) 
            if z.size(0) > 1:
                z = self.bn[i](z)
        z_mu = torch.relu(self.network[-1](z))

        return z_mu

class Phi(nn.Module):
    """
    Phi: x -> phi(x) in R^d
    """
    def __init__(
        self, 
        x_dim,
        network_dims=[256, 256, 256],
        use_image_feature=False
        ):

        super(Phi, self).__init__()

        if use_image_feature:
            self.image_feature = Image_Feature(num_dense_feature=0).to(device)
            input_dim = self.image_feature.num_dense_feature + self.image_feature.linear.out_features
        else:
            input_dim = x_dim

        network_dims = [input_dim] + network_dims
        
        # create network list using network_dims
        self.network = nn.ModuleList()
        self.bn = nn.ModuleList()

        for i in range(len(network_dims)-1):
            self.network.append(nn.Linear(network_dims[i], network_dims[i+1]))
            self.bn.append(nn.BatchNorm1d(network_dims[i+1]))

        self.use_image_feature = use_image_feature

    def forward(self, x):
        x = self.image_feature(x) if self.use_image_feature else x
        for i in range(len(self.network)-1):
            x = F.relu(self.network[i](x)) 
            if x.size(0) > 1:
                x = self.bn[i](x)
        x_phi = torch.tanh(self.network[-1](x))

        return x_phi

class Xi(nn.Module):
    """
    Xi: o -> xi(o) in R^d
    """
    def __init__(
        self, 
        o_dim,
        network_dims=[256, 256, 256],
        use_image_feature=False
        ):

        super(Xi, self).__init__()

        if use_image_feature:
            self.image_feature = Image_Feature(num_dense_feature=1).to(device)
            input_dim = self.image_feature.num_dense_feature + self.image_feature.linear.out_features
        else:
            input_dim = o_dim

        network_dims = [input_dim] + network_dims
        
        # create network list using network_dims
        self.network = nn.ModuleList()
        self.bn = nn.ModuleList()

        for i in range(len(network_dims)-1):
            self.network.append(nn.Linear(network_dims[i], network_dims[i+1]))
            self.bn.append(nn.BatchNorm1d(network_dims[i+1]))

        self.use_image_feature = use_image_feature

    def forward(self, o):
        o = self.image_feature(o) if self.use_image_feature else o
        for i in range(len(self.network)-1):
            o = F.relu(self.network[i](o)) 
            if o.size(0) > 1:
                o = self.bn[i](o)
        o_xi = torch.tanh(self.network[-1](o))

        return o_xi

class Nu(nn.Module):
    """
    Nu: y -> nu(y) in R^d
    """
    def __init__(
        self, 
        y_dim,
        network_dims=[256, 256, 256],
        use_image_feature=False
        ):

        super(Nu, self).__init__()

        if use_image_feature:
            self.image_feature = Image_Feature(num_dense_feature=2).to(device)
            input_dim = self.image_feature.num_dense_feature + self.image_feature.linear.out_features
        else:
            input_dim = y_dim

        network_dims = [input_dim] + network_dims

        # create network list using network_dims
        self.network = nn.ModuleList()
        self.bn = nn.ModuleList()

        for i in range(len(network_dims)-1):
            self.network.append(nn.Linear(network_dims[i], network_dims[i+1]))
            self.bn.append(nn.BatchNorm1d(network_dims[i+1]))
    
        self.use_image_feature = use_image_feature

    def forward(self, y):
        y = self.image_feature(y) if self.use_image_feature else y
        for i in range(len(self.network)-1):
            y = F.relu(self.network[i](y)) 
            if y.size(0) > 1:
                y = self.bn[i](y)
        y_nu = torch.relu(self.network[-1](y))

        return y_nu
    
