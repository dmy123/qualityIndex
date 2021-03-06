import torch.nn.functional as F
import scipy.io as scio
import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)

################################################################
# fourier layers
# 训练生成温度场神经网络模型
################################################################

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

    def forward(self, x):   #20*20*32*32*24
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 3, normalized=True, onesided=True)    # 20*20*32*32*13*2

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)   # 20*20*32*32*13*2
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)    # weights1-4:20*20*12*12*4*2
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
    def __init__(self, modes1, modes2, modes3, width):   #12*12*4*20
        # self.conv表示卷积层
        # self.fc表示线性层，nn.Linear(param1, param2)输入param1个特征，输出param2个特征
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(4, self.width)

        self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # nn.Conv1d(输入信号通道，卷积产生通道，卷积核的尺寸)，该方法表示一维卷积
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        # torch.nn.BatchNorm3d：对4d数据组成的5d输入进行BN，在训练时，该层计算每次输入的均值和方差，并进行平行移动。移动平均默认的动量为0.
        #在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):  # 20*32*32*24*4
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)  #20*32*32*24*20
        x = x.permute(0, 4, 1, 2, 3)  #20*20*32*32*24

        x1 = self.conv0(x)            #20*20*32*32*24
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)    #20*20*32*32*24
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)   # 20*32*32*24*128
        x = F.relu(x)
        x = self.fc2(x)   # 20*32*32*24*1
        return x

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        self.conv1 = SimpleBlock2d(modes, modes, 4, width)


    def forward(self, x):
        x = self.conv1(x)
        return x


    def count_params(self):
        c = 0
        # 20*20*12*12*4*2,20*20*1,20
        for p in self.parameters():   #p:20*4
            c += reduce(operator.mul, list(p.size()))

        return c

################################################################
# configs
################################################################
TRAIN_PATH = 'data/temperature_data.mat'
TEST_PATH = 'data/temperature_data.mat'

#训练数据个数与测试数据个数
ntrain = 160
ntest = 40

modes = 12
width = 20

batch_size = 20
batch_size2 = batch_size


epochs = 500
learning_rate = 0.0025
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

# path = 'ns_fourier_3d_rnn_V10000_T20_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
# path_model = 'model/'+path
# path_train_err = 'results/'+path+'train.txt'
# path_test_err = 'results/'+path+'test.txt'
# path_image = 'image/'+path
# path = 'temperature_field_240_32_32_1920'+'_2021_1_20'
path = 'temperature_field_240_16_16_96'+'_2021_2_02'
path_model = 'model/'+path


runtime = np.zeros(2, )
t1 = default_timer()


# sub = 8
# S = 4
sub = 1
S = 16
T_in = 48
T_start = 0
step = T_in - T_start
T = 48

################################################################
# load data
################################################################
data = scio.loadmat(TRAIN_PATH)
print(data.keys())
print(type(data['temperature_field_data']))
print(data['temperature_field_data'].shape)

# reader = MatReader(TRAIN_PATH)
# train_a = reader.read_field('temperature_field_data')[:ntrain,::sub,::sub,T_start:T_in*20:20]
# train_u = reader.read_field('temperature_field_data')[:ntrain,::sub,::sub,T_in*20:(T+T_in)*20:20]
#
# reader = MatReader(TEST_PATH)
# test_a = reader.read_field('temperature_field_data')[-ntest:,::sub,::sub,T_start:T_in*20:20]
# test_u = reader.read_field('temperature_field_data')[-ntest:,::sub,::sub,T_in*20:(T+T_in)*20:20]
reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('temperature_field_data')[:ntrain,::sub,::sub,T_start:T_in]
train_u = reader.read_field('temperature_field_data')[:ntrain,::sub,::sub,T_in:(T+T_in)]

reader = MatReader(TEST_PATH)
test_a = reader.read_field('temperature_field_data')[-ntest:,::sub,::sub,T_start:T_in]
test_u = reader.read_field('temperature_field_data')[-ntest:,::sub,::sub,T_in:(T+T_in)]
print('test_a.shape',test_a.shape)
print(train_u.shape, test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])



train_a = train_a.reshape(ntrain,S,S,step,1)
test_a = test_a.reshape(ntest,S,S,step,1)
print('test_a.shape',test_a.shape)
# cat the location information (x,y,t)
gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, step, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, step, 1])
gridt = torch.tensor(np.linspace(0, 1, step+1)[1:], dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, step, 1).repeat([1, S, S, 1, 1])

train_a = torch.cat((gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
                       gridt.repeat([ntrain,1,1,1,1]), train_a), dim=-1)
test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                       gridt.repeat([ntest,1,1,1,1]), test_a), dim=-1)

#print(train_a.shape, train_u.shape)
print("train_a.shape",train_a.shape)
print("train_u.shape",train_u.shape)
print("test_a.shape",test_a.shape)
print("test_u.shape",test_u.shape)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

################################################################
# training and evaluation
################################################################
model = Net2d(modes, width).cuda()
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

print(model.count_params())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)

gridx = gridx.to(device)
gridy = gridy.to(device)
gridt = gridt.to(device)

train_relative_error = np.empty((epochs))
test_relative_error = np.empty((epochs))

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t+step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size,-1), y.reshape(batch_size,-1))  #loss是个scalar，我们可以直接用item获取到他的python类型的数值

            if t == 0:
                pred = im.squeeze()
            else:
                pred = torch.cat((pred, im.squeeze()), -1)

            im = torch.cat((gridx.repeat([batch_size, 1, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1, 1]),
                                 gridt.repeat([batch_size, 1, 1, 1, 1]), im), dim=-1)
            xx = torch.cat([xx[..., step:, :], im], -2)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        # l2_full.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im.squeeze()
                else:
                    pred = torch.cat((pred, im.squeeze()), -1)

                im = torch.cat((gridx.repeat([batch_size, 1, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1, 1]),
                                gridt.repeat([batch_size, 1, 1, 1, 1]), im), dim=-1)
                xx = torch.cat([xx[..., step:, :], im], -2)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    train_relative_error[ep] = train_l2_step/ntrain/(T/step)
    test_relative_error[ep] = test_l2_step/ntest/(T/step)
    print(ep, t2-t1, train_l2_step/ntrain/(T/step), train_l2_full/ntrain, test_l2_step/ntest/(T/step), test_l2_full/ntest)
x_axis = np.arange(epochs)
plt.axes(yscale = "log")
plt.plot(x_axis,train_relative_error)
plt.xlabel('Epochs')
plt.ylabel('Relative error')
plt.savefig('picture/train_relative_error.jpg')
plt.show()
plt.axes(yscale = "log")
plt.plot(x_axis,test_relative_error)
plt.xlabel('Epochs')
plt.ylabel('Relative error')
plt.savefig('picture/test_relative_error.jpg')
plt.show()
torch.save(model, path_model)


# pred = torch.zeros(test_u.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# with torch.no_grad():
#     for x, y in test_loader:
#         test_l2 = 0;
#         x, y = x.cuda(), y.cuda()
#
#         out = model(x)
#         out = y_normalizer.decode(out)
#         pred[index] = out
#
#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         print(index, test_l2)
#         index = index + 1

# scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})




