import torch

'''
Even though implementing EuDist2() here, we prefer the alternative 
torch.cdist() which is optimized and should be faster.
  torch.cdist(fea_a, fea_b) = EuDist2(fea_a, fea_b)
  torch.cdist(fea_a, fea_a) = EuDist2(fea_a)
which calculates the pair-wise euclidean distance between each row of fea_a and fea_b
or between each row of fea_a and itself.
Please refer to https://pytorch.org/docs/stable/generated/torch.cdist.html for more details.
'''

def EuDist2(fea_a, fea_b=None, bSqrt=True):
    if fea_b is None:
        aa = torch.sum(fea_a.square(), dim=1).view(-1, 1)
        ab = fea_a @ fea_a.t()
        D = aa + aa.t() - 2 * ab
    else:
        aa = torch.sum(fea_a.square(), dim=1).view(-1, 1)
        bb = torch.sum(fea_b.square(), dim=1).view(1, -1)
        ab = fea_a @ fea_b.t()
        D = aa + bb - 2 * ab

    # Clamp values less than 0 to 0
    D[D < 0] = 0
    
    if bSqrt:
        D = torch.sqrt(D)
    
    if fea_b is None:
        # Make sure the resulting distance matrix is symmetric
        D = torch.max(D, D.t())

    return D


