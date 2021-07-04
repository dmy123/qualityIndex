import numpy as np
import torch.nn.functional as F
from utilities3 import *
import operator
from functools import reduce
from functools import partial
from crackAndStress.stress import solver
from crackAndStress.crack import crack
from temperatureModel.temperatureField.temperatureFieldCal import one_example_temp_cal
from temperatureModel.temperatureField.temperature_cal import constrain_punish
from GetOneTemperatureField import getOneTempField
from pakage import continuousCastVariable
import time

# 计算温度场和应力场及相应指标
# 计算损失函数

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

def tempCrackAndStress(var_h_initial,v_cast,t_cast,TraditionNotFno,model):
    ccv = continuousCastVariable.ContinuousCastVariable(var_h_initial, v_cast, t_cast)
    start_time = time.time()
    if (TraditionNotFno):
        # print('温度场不加速')
        MiddleTemp_all22, t = one_example_temp_cal(ccv)
    else:
        # print('温度场加速')
        MiddleTemp_all22, t = getOneTempField(ccv,model)
    end_time = time.time()
    cal_time = end_time - start_time
    # print('温度场计算时间：', cal_time)
    # print('MiddleTemp_all22',type(MiddleTemp_all22))
    # print('t', type(t))
    start_time = time.time()
    result_crack, crack_list = crack(MiddleTemp_all22, ccv.var_XNumber, ccv.var_X, ccv.var_YNumber, ccv.var_Y, ccv.Time_all, ccv.var_liqTemp,
                                     ccv.var_SodTemp)  # 缩孔
    end_time = time.time()
    cal_time = end_time - start_time
    # print('缩孔计算时间：', cal_time)
    start_time = time.time()
    stress, estress, strain, f2 = solver(ccv.var_XNumber, ccv.var_YNumber, ccv.var_X/ccv.var_XNumber, ccv.var_Y/ccv.var_YNumber, t, int(ccv.t_l[-3]), int(ccv.t_l[-1]))  # 计算节点位移
    stress, estress1, strain, f3 = solver(ccv.var_XNumber, ccv.var_YNumber, ccv.var_X/ccv.var_XNumber, ccv.var_Y/ccv.var_YNumber, t, int(ccv.t_l[-2]), int(ccv.t_l[-1]))  # 计算节点位移
    stress, estress1, strain, f4 = solver(ccv.var_XNumber, ccv.var_YNumber, ccv.var_X/ccv.var_XNumber, ccv.var_Y/ccv.var_YNumber, t, int(ccv.t_l[-1]), int(ccv.t_l[-1]))  # 计算节点位移
    stress, estress1, strain, f1 = solver(ccv.var_XNumber, ccv.var_YNumber, ccv.var_X/ccv.var_XNumber, ccv.var_Y/ccv.var_YNumber, t, int(ccv.t_l[-4]), int(ccv.t_l[-1]))  # 计算节点位移
    end_time = time.time()
    cal_time = end_time - start_time
    # print('裂纹计算时间：', cal_time)
    start_time = time.time()
    mental_punish = constrain_punish(np.array(t), ccv.var_dis, ccv.t_l, ccv.time_Mold, ccv.var_XNumber, ccv.var_YNumber, ccv.var_SodTemp,
                                     ccv.strand_shell_set_loc, ccv.max_surface_temperature, ccv.min_surface_temperature, ccv.temp_rise,
                                     ccv.D_T_uplim, ccv.D_T_downlim, ccv.straighten_point, ccv.Tj_min, ccv.Tj_max)
    end_time = time.time()
    cal_time = end_time - start_time
    # print('冶金准则计算时间：', cal_time)

    # stress, estress, strain, f2 = solver(20, 20, 0.1, 0.1, t, int(t_l[-3]), int(t_l[-1]))  # 计算节点位移
    # stress, estress1, strain, f3 = solver(20, 20, 0.1, 0.1, t, int(t_l[-2]), int(t_l[-1]))  # 计算节点位移
    # stress, estress1, strain, f4 = solver(20, 20, 0.1, 0.1, t, int(t_l[-1]), int(t_l[-1]))  # 计算节点位移
    # stress, estress1, strain, f1 = solver(20, 20, 0.1, 0.1, t, int(t_l[-4]), int(t_l[-1]))  # 计算节点位移
    return f1 / 7, result_crack, f2, f3, mental_punish, t, crack_list, strain, estress, f4 / 7

def lossFunction(var_h_initial,v_cast,t_cast,TraditionNotFno,model):
    start_time = time.time()
    a,b,c,f,g,t,crack_list,strain,estress,f4 = tempCrackAndStress(var_h_initial, v_cast, t_cast,TraditionNotFno,model)
    end_time = time.time()
    cal_time = end_time - start_time
    # print("计算一次lossFunction：", cal_time)
    # print(a, b, c, f, g, np.array(t).shape, np.array(crack_list).shape,
    #       np.array(strain).shape, np.array(estress).shape, f4)
    return 0.025*a+0.28*c+0.28*f+0.198*f4+0.236*b+50*g[-1]
    # a, b, c, f, t, crack_list, strain, estress, f4 = tempCrackAndStredd(var_h_initial, v_cast, t_cast)
    # return 0.025*a+0.28*c+0.28*f+0.198*f4+0.236*b

# var_h_initial=[63.65,138.36,100.87,80.363]

var_h_initial=[25.5,28.5,9.15,6.8]
v_cast = 0.6
t_cast = 1530
TraditionNotFno = 1    #温度场不加速
# TraditionNotFno = 0  #温度场加速
# model = torch.load('/tmp/pycharm_project_82/temperature_field_240_16_16_96_2021_2_02')
model = torch.load('/tmp/pycharm_project_82/temperatureModel/model/temperature_field_240_16_16_96_2021_2_02')
print(lossFunction(var_h_initial,v_cast,t_cast,TraditionNotFno,model))