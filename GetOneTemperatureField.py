import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utilities3 import *
import operator
from functools import reduce
from functools import partial
import time
from timeit import default_timer
import scipy.io
from temperatureModel.temperatureField.temperatureFieldCal import onetempfieldcal

torch.manual_seed(0)
np.random.seed(0)

def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

class SpectralConv3d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 3, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.irfft(out_ft, 3, normalized=True, onesided=True, signal_sizes=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        # print("-->{}".format(width.size()))
        self.fc0 = nn.Linear(4, self.width)

        self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        #print("-->{}".format(x.size()))
        x = x.permute(0, 4, 1, 2, 3)
        # print("-->{}".format(x.size()))
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        # print("-->{}".format(x.size()))
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        #print("-->{}".format(x.size()))
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        #print("-->{}".format(x.size()))
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn3(x1 + x2)
        #print("-->{}".format(x.size()))


        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        #print("-->{}".format(x.size()))
        x = F.relu(x)
        x = self.fc2(x)
        #print("-->{}".format(x.size()))
        return x

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        self.conv1 = SimpleBlock2d(modes, modes, 6, width)


    def forward(self, x):
        x = self.conv1(x)
        #print("-->{}".format(x.size()))
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

def getOneTempField(ccv):
    t1 = default_timer()
    MiddleTemp_all = onetempfieldcal(ccv)

    ntest = 1

    # sub = 4
    # sub_t = 4
    sub = 1
    sub_t = 1
    S = 32
    T_in = ccv.Time_all_a
    T = ccv.Time_all - ccv.Time_all_a

    indent = 3

    test_a = np.zeros((1,ccv.var_XNumber,ccv.var_YNumber,ccv.Time_all_a))
    test_a[0] = MiddleTemp_all
    test_u = np.zeros((1,ccv.var_XNumber,ccv.var_YNumber,ccv.Time_all-ccv.Time_all_a))
    test_a = torch.from_numpy(test_a).float()
    test_u = torch.from_numpy(test_u).float()
    res_a = test_a

    # print(test_a.shape, test_u.shape)

    # pad the location information (s,t)
    S = S * (1//sub)
    T = T * (1//sub_t)

    test_a = test_a.reshape(ntest,S,S,T,1)
    # print(test_a.shape)

    gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])

    test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                           gridt.repeat([ntest,1,1,1,1]),
                        test_a),
                       dim=-1)

    t2 = default_timer()
    # print('preprocessing finished, time used:', t2-t1)
    device = torch.device('cuda')

    # modes = 12
    # width = 20

    # load model
    # model = torch.load('/tmp/pycharm_project_82/temperatureModel/model/temperature_field_240_32_32_1920_2021_1_11')
    model = torch.load('/tmp/pycharm_project_82/temperature_field_240_32_32_1920_2021_1_11')

    # test
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1,
                                              shuffle=False)
    pred = torch.zeros(test_u.shape)
    index = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            start_time = time.time()
            out = model(x)
            pred[index] = out
            end_time = time.time()
            cal_time = end_time - start_time
            print("计算输出温度场时间：", cal_time)
            index = index + 1

    # path = 'one_temperature_field_eval_pred_u'
    # scipy.io.savemat(path+'.mat', mdict={'pred': pred.cpu().numpy(), 'u': test_u.cpu().numpy()})
    path = 'one_temperature_field_result'
    res_MiddleTemp_all = np.concatenate((res_a,pred.cpu().numpy()),axis = 3)
    res_t= np.zeros((res_MiddleTemp_all.shape[0],res_MiddleTemp_all.shape[3],res_MiddleTemp_all.shape[1],res_MiddleTemp_all.shape[2]))
    # print(res_MiddleTemp_all.shape)
    # 转换
    for stepTime in range(ccv.Time_all):
        for i in range(ccv.var_XNumber):
            for j in range(ccv.var_YNumber):
                res_t[0][stepTime][i][j] = res_MiddleTemp_all[0][i][j][stepTime]
    # scipy.io.savemat(path+'.mat', mdict={'temperature_field_result_res_MiddleTemp_all':res_MiddleTemp_all,'temperature_field_result_res_t':res_t})
    return res_MiddleTemp_all[0],res_t[0]


# cost = 0
# for time in range(960):
#     for x in range(S):
#         for y in range(S):
#             cost = cost + pow(pred[0][x][y][time] -test_u[0][x][y][time],2)
# print('花费：',cost/960)