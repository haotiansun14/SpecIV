import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Image_Feature(nn.Module):

    def __init__(self, num_dense_feature: int):
        super(Image_Feature, self).__init__()
        self.num_dense_feature = num_dense_feature
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.linear = nn.Linear(32 * 7 * 7, 1024)

    def forward(self, data):
        dense = data[:, :self.num_dense_feature]
        image = data[:, self.num_dense_feature:]
        x = image.reshape((-1, 1, 28, 28))
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.linear(x)
        return torch.cat([dense, output], dim=1)
