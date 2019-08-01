import torch
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):
    def __init__(self, input_shape):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10*3*3, 32),
#             nn.Tanh(),
#             nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
#             nn.Tanh()
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))

    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.localization(x)
#         xs = self.globalAvgPool(xs)
#         print(xs.shape)
        xs = xs.view(xs.shape[0], -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
#         print(theta)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x





# import os 
# import numpy as np 
# import torch 
# import torch.nn as nn 
# import torch.nn.functional as F 

# class STN(nn.Module):
#     """
#     Implements a spatial transformer 
#     as proposed in the Jaderberg paper. 
#     Comprises of 3 parts:
#     1. Localization Net
#     2. A grid generator 
#     3. A roi pooled module.
#     The current implementation uses a very small convolutional net with 
#     2 convolutional layers and 2 fully connected layers. Backends 
#     can be swapped in favor of VGG, ResNets etc. TTMV
#     Returns:
#     A roi feature map with the same input spatial dimension as the input feature map. 
#     """
#     def __init__(self, in_channels, spatial_dims, kernel_size,use_dropout=False):
#         super(STN, self).__init__()
#         self._h, self._w = spatial_dims 
#         self._in_ch = in_channels 
#         self._ksize = kernel_size
#         self.dropout = use_dropout

#         # localization net 
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False) # size : [1x3x32x32]
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=self._ksize, stride=1, padding=1, bias=False)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=self._ksize, stride=1, padding=1, bias=False)

#         self.fc1 = nn.Linear(64, 1024)
#         self.fc2 = nn.Linear(1024, 6)
#         self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))

#     def forward(self, x): 
#         """
#         Forward pass of the STN module. 
#         x -> input feature map 
#         """
#         batch_images = x 
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x,2)
#         x = F.relu(self.conv4(x))
# #         x = F.max_pool2d(x, 2)
#         x = self.globalAvgPool(x)
# #         print("Pre view size:{}".format(x.size()))
#         x = x.view(x.shape[0], -1)
#         if self.dropout:
#             x = F.dropout(self.fc1(x), p=0.5)
#             x = F.dropout(self.fc2(x), p=0.5)
#         else:
#             x = self.fc1(x)
#             x = torch.tanh(self.fc2(x)) # params [Nx6]
        
#         x = x.view(-1, 2,3) # change it to the 2x3 matrix 
# #         print(x.size())
#         affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
# #         print(affine_grid_points.shape)
#         assert(affine_grid_points.size(0) == batch_images.size(0)), "The batch sizes of the input images must be same as the generated grid."
#         rois = F.grid_sample(batch_images, affine_grid_points)
# #         print("rois found to be of size:{}".format(rois.size()))
#         return rois, affine_grid_points






# class BoundedGridLocNet(nn.Module):

#     def __init__(self, grid_height, grid_width, target_control_points):
#         super(BoundedGridLocNet, self).__init__()
#         self.cnn = CNN(grid_height * grid_width * 2)

#         bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
#         bias = bias.view(-1)
#         self.cnn.fc2.bias.data.copy_(bias)
#         self.cnn.fc2.weight.data.zero_()

#     def forward(self, x):
#         batch_size = x.size(0)
#         points = F.tanh(self.cnn(x))
#         return points.view(batch_size, -1, 2)

# class UnBoundedGridLocNet(nn.Module):

#     def __init__(self, grid_height, grid_width, target_control_points):
#         super(UnBoundedGridLocNet, self).__init__()
#         self.cnn = CNN(grid_height * grid_width * 2)

#         bias = target_control_points.view(-1)
#         self.cnn.fc2.bias.data.copy_(bias)
#         self.cnn.fc2.weight.data.zero_()

#     def forward(self, x):
#         batch_size = x.size(0)
#         points = self.cnn(x)
# return points.view(batch_size, -1, 2)

# class STNClsNet(nn.Module):

#     def __init__(self, input_shape):
#         super(STNClsNet, self).__init__()
#         self.input_shape = input_shape

#         r1 = 0.9
#         r2 = 0.9
#         assert r1 < 1 and r2 < 1 # if >= 1, arctanh will cause error in BoundedGridLocNet
#         target_control_points = torch.Tensor(list(itertools.product(
#             np.arange(-r1, r1 + 0.00001, 2.0  * r1 / (4 - 1)),
#             np.arange(-r2, r2 + 0.00001, 2.0  * r2 / (4 - 1)),
#         )))
#         Y, X = target_control_points.split(1, dim = 1)
#         target_control_points = torch.cat([X, Y], dim = 1)

#         GridLocNet = UnBoundedGridLocNet
# #         GridLocNet = BoundedGridLocNet
        
#         self.loc_net = GridLocNet(4, 4, target_control_points)

#         self.tps = TPSGridGen(input_shape[1], input_shape[2], target_control_points)

#         self.cls_net = ClsNet()

#     def forward(self, x):
#         batch_size = x.size(0)
#         source_control_points = self.loc_net(x)
#         source_coordinate = self.tps(source_control_points)
#         grid = source_coordinate.view(batch_size, self.input_shape[1], self.input_shape[2], 2)
#         transformed_x = F.grid_sample(x, grid)
# #         logit = self.cls_net(transformed_x)
#     return transformed_x