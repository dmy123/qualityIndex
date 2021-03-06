import torch.nn.functional as F
from utilities3 import *
import operator
from functools import reduce
from functools import partial
import time
from timeit import default_timer
import scipy.io as scio

# 测试用
# 预测数据并与实际数据对比

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

t1 = default_timer()

TEST_PATH = 'data/temperature_data.mat'

ntest = 40

# sub = 4
sub = 1
# sub_t = 4
sub_t = 3
# S = 32
# T_in = 480
# T = 1440
S = 16
T_in = 48
T = 96

indent = 3

# load data
reader = MatReader(TEST_PATH)
test_a = reader.read_field('temperature_field_data')[160:200:1,::sub,::sub, 0:T_in] #([0, T_in])
test_u = reader.read_field('temperature_field_data')[160:200:1,::sub,::sub, T_in:T_in + T] #([T_in, T_in + T])
res_a = test_a
print('type(test_a)',type(test_a))
print(test_a.shape, test_u.shape)

# pad the location information (s,t)
# S = S * (4//sub)
# T = T * (4//sub_t)
S = S * (1//sub)
T = T * (1//sub_t)
T = T_in
#test_a = test_a.reshape(ntest,S,S,1,T_in).repeat([1,1,1,T,1])
test_a = test_a.reshape(ntest,S,S,T,1)
print(test_a.shape)

gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])

test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                       gridt.repeat([ntest,1,1,1,1]), test_a), dim=-1)


t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

# modes = 12
# width = 20

# load model
model = torch.load('model/temperature_field_240_16_16_96_2021_2_02')
# model = Net2d(modes, width).cuda()

print(model.count_params())

# test
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
print("test_a.shape",test_a.shape)
print("test_u.shape",test_u.shape)
myloss = LpLoss(size_average=False)
pred = torch.zeros(test_u.shape)
index = 0
cost_time = 0
with torch.no_grad():
    test_l2 = 0
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        start_time = time.time()
        out = model(x)
        print(out.size())
        pred[index] = out
        end_time = time.time()
        cal_time = end_time - start_time
        print("计算一次温度场时间：", cal_time)
        cost_time = cost_time + cal_time

        loss = myloss(out.view(1, -1), y.view(1, -1)).item()
        test_l2 += loss
        print(index, loss)
        index = index + 1
print(test_l2/ntest)
print("温度场计算总时间：",cost_time,",单次温度场计算平均时间：",cost_time / ntest)
print('pred:',type(pred))
print('pred:',type(pred.cpu()))
print('pred:',type(pred.cpu().numpy()))
print(res_a.shape)
print(pred.cpu().numpy().shape)
result = np.concatenate((res_a,pred.cpu().numpy()),axis = 3)
print(result.shape)
path = 'temperature_field_eval'
scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy(), 'u': test_u.cpu().numpy()})
# path= 'temperature_field_result'
# scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy(), 'u': test_u.cpu().numpy()})





