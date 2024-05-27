import torch

def k_prod(mat1: torch.Tensor, mat2: torch.Tensor, mode):
    mat1_shape = tuple(mat1.size())
    mat2_shape = tuple(mat2.size())
    nData = mat2_shape[0]
    
    if mode == 'last':
        if mat1.dim() == 3:
            mat1 = mat1.unsqueeze(0).expand(nData, -1, -1, -1)
        return torch.einsum('bijk,bk->bij', mat1, mat2)
    
    elif mode == 'all':
        assert mat1_shape[0] == mat2_shape[0], f"mat1_shape[0] = {mat1_shape[0]}, mat2_shape[0] = {mat2_shape[0]}"
        aug_mat1_shape = mat1_shape + (1,) * (len(mat2_shape) - 1)
        aug_mat1 = torch.reshape(mat1, aug_mat1_shape)
        aug_mat2_shape = (nData,) + (1,) * (len(mat1_shape) - 1) + mat2_shape[1:]
        aug_mat2 = torch.reshape(mat2, aug_mat2_shape)
        return (aug_mat1 * aug_mat2).flatten(start_dim=1)
    
    else:
        raise ValueError('mode not recognized')
    

