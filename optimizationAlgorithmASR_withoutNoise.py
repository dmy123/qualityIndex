import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
from lossFunction import lossFunction
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)

from utilities3 import *
import operator
from functools import reduce
from functools import partial


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

def main():
    """
    问题二
    dimension = 2
    feasible_region = np.array([[0, 50], [0, 50]])
    r = np.array([1, 1])
    ASR(dimension, feasible_region, r) """
    """
    连铸
    """
    dimension = 4
    feasible_region = np.array([[8, 13], [35, 45], [20, 25], [12, 17]])
    r = np.array([0.1, 0.2, 0.1, 0.1])
    test_times = 2
    for i in range(test_times):
        print("第",i+1,"次实验")
        ASR(dimension, feasible_region, r)
    return 0


def ASR(dimension, feasible_region, r):
    """参数定义"""
    b = 1.1
    c = 0.5
    Ck = 1
    g = 0.5
    delta = 0.01
    K = 200
    T = 0.1
    """参数定义"""

    i = 1
    k = 0
    # iterations = 1000001
    # iterations = 800
    iterations = 200
    accepted_count = 1
    simulation_budget = 1

    accepted_set = np.zeros((20000, dimension + 1))
    accepted_count = 0
    best_solution = np.zeros(dimension + 1)
    best_solution_loc = 0
    best_solution_change = np.zeros((100, dimension + 1))
    best_solution_change_count = 0
    paint = np.zeros((100, 3))

    paint_noise = np.zeros(iterations + 1)

    start = time.time()

    model = torch.load('/tmp/pycharm_project_82/temperatureModel/model/temperature_field_240_16_16_96_2021_2_02')
    # model = torch.load('/tmp/pycharm_project_82/temperature_field_240_32_32_1920_2021_1_11')

    while (k <= iterations):
        k = k + 1
        # print("迭代次数：",k)
        new_sample = np.zeros(dimension)
        if (uniformly_distribution(0, 1) < g or i == 1):
            for dim in range(dimension):
                accepted_set[accepted_count][dim] = uniformly_distribution(feasible_region[dim][0],
                                                                           feasible_region[dim][1])
                new_sample[dim] = accepted_set[accepted_count][dim]
        else:
            for dim in range(dimension):
                local_min = max(best_solution[dim] - r[dim], feasible_region[dim][0])
                local_max = min(best_solution[dim] + r[dim], feasible_region[dim][1])
                accepted_set[accepted_count][dim] = uniformly_distribution(local_min, local_max)
                new_sample[dim] = accepted_set[accepted_count][dim]

        # new_sample_value = test_2_twohills(new_sample)
        new_sample_value, paint_noise[k - 1] = function(new_sample, model)
        new_sample_value = -new_sample_value
        accepted_set[accepted_count][dimension] = new_sample_value

        if (i == 1):
            for dim in range(dimension + 1):
                best_solution[dim] = accepted_set[accepted_count][dim]
                best_solution_change[best_solution_change_count][dim] = accepted_set[accepted_count][dim]

            estimated_value = {accepted_count: [new_sample_value, 1]}
            # paint[best_solution_change_count] = [simulation_budget, test_2_twohills_withoutnoise(new_sample) , new_sample_value]
            # paint[best_solution_change_count] = [simulation_budget, -function(new_sample, model), new_sample_value]
            paint[best_solution_change_count] = [simulation_budget, new_sample_value, new_sample_value]
            accepted_count = accepted_count + 1
            best_solution_change_count = best_solution_change_count + 1
        else:
            # count = K - 1
            # add = new_sample_value
            # while (count > 0):
            #     # add += test_2_twohills(new_sample)
            #     add += -function(new_sample, model)
            #     count = count - 1
            #
            # simulation_budget += K
            # new_sample_value = add / K
            if (new_sample_value >= best_solution[dimension] - delta):
                simulation_budget = simulation_budget + 1
                if (accepted_set[accepted_count][dimension] > best_solution[dimension]):
                    for dim in range(dimension + 1):
                        best_solution[dim] = accepted_set[accepted_count][dim]
                        best_solution_change[best_solution_change_count][dim] = accepted_set[accepted_count][dim]

                    # paint[best_solution_change_count] = [simulation_budget,  test_2_twohills_withoutnoise(accepted_set[t][:dimension]) ,best_solution[dimension]]
                    paint[best_solution_change_count] = [simulation_budget,
                                                         best_solution[dimension],
                                                         best_solution[dimension]]
                    best_solution_loc = accepted_count
                    best_solution_change_count = best_solution_change_count + 1
                estimated_value[accepted_count] = [new_sample_value, 1]  # 要改吗
                accepted_count = accepted_count + 1



        # for t in range(accepted_count):
        #     if (estimated_value[t][1] < rise_rate_K(i, c, Ck)):
        #         additional_ob_times = rise_rate_K(i, c, Ck) - estimated_value[t][1]
        #         additional_observe = 0
        #         simulation_budget = simulation_budget + additional_ob_times
        #         estimated_value[t][1] = estimated_value[t][1] + additional_ob_times
        #         while (additional_ob_times > 0):
        #             # additional_observe = additional_observe + test_2_twohills(accepted_set[t][:dimension])
        #             additional_observe = additional_observe - function(accepted_set[t][:dimension], model)
        #             additional_ob_times = additional_ob_times - 1
        #
        #         estimated_value[t][0] = estimated_value[t][0] + additional_observe
        #         accepted_set[t][dimension] = estimated_value[t][0] / estimated_value[t][1]
        #         if (t == best_solution_loc):
        #             best_solution[dimension] = accepted_set[t][dimension]
        #         if (accepted_set[t][dimension] > best_solution[dimension]):
        #             for dim in range(dimension + 1):
        #                 best_solution[dim] = accepted_set[t][dim]
        #                 best_solution_change[best_solution_change_count][dim] = accepted_set[t][dim]
        #
        #             # paint[best_solution_change_count] = [simulation_budget,  test_2_twohills_withoutnoise(accepted_set[t][:dimension]) ,best_solution[dimension]]
        #             paint[best_solution_change_count] = [simulation_budget,
        #                                                  -function(accepted_set[t][:dimension], model),
        #                                                  best_solution[dimension]]
        #             best_solution_loc = t
        #             best_solution_change_count = best_solution_change_count + 1
        i = i + 1

    end = time.time()
    # print(probability)
    print("Execution Time", end - start)
    print("接收点集合数量", accepted_count)
    print("最优点变化数量", best_solution_change_count)
    print("best_solution变化：", best_solution_change[:best_solution_change_count])
    print("best_solution", best_solution)
    print("paint", paint[:best_solution_change_count])
    # painting = np.array(paint[:best_solution_change_count][:2])
    print(paint[:best_solution_change_count, 0])
    print(paint[:best_solution_change_count, 2])
    # plt.plot(paint[:best_solution_change_count, 0],paint[:best_solution_change_count, 1])
    plt.plot(paint[:best_solution_change_count, 0], -paint[:best_solution_change_count, 2], color='orangered',
             linewidth=1)
    plt.scatter(paint[:best_solution_change_count, 0], -paint[:best_solution_change_count, 2], color='orangered',
                marker="*")
    plt.xlabel('number of objective function evaluations')
    plt.ylabel('the optimal solution')
    plt.show()
    plt.plot(np.arange(best_solution_change_count), best_solution_change[:best_solution_change_count, 0])
    plt.plot(np.arange(best_solution_change_count), best_solution_change[:best_solution_change_count, 1])
    plt.plot(np.arange(best_solution_change_count), best_solution_change[:best_solution_change_count, 2])
    plt.plot(np.arange(best_solution_change_count), best_solution_change[:best_solution_change_count, 3])
    plt.xlabel('iteration times')
    plt.ylabel('the optimal solution')
    plt.show()
    plt.plot(np.arange(iterations + 1), paint_noise)
    plt.xlabel('time')
    plt.ylabel('speed')
    plt.show()
    # plt.xlabel("目标函数计算次数")
    # plt.ylabel("最优适应值")
    # plt.xlabel(u'目标函数计算次数', fontproperties=font_set)
    # plt.ylabel(u'最优适应值', fontproperties=font_set)
    plt.show()
    return 0


def M(i, b):
    return math.floor(math.pow(i, b))


def rise_rate_K(k, c, Ck):
    return math.floor(Ck * math.pow(k, c) + 1)


def Tk_(T, k_):
    return T / math.log(k_ + 1)


def uniformly_distribution(a, b):
    return random.uniform(a, b)


def find_resample(resample, probability):
    begin = 0
    end = probability.size
    while (begin < end):
        mid = int(begin + (end - begin) / 2)
        if (probability[mid] > resample):
            end = mid
        elif (probability[mid] < resample):
            begin = mid + 1
        else:
            return mid

    if (begin == 0):
        return 0
    return begin - 1;


def float_equal(a, b):
    return math.inclose(a, b, rel_tol=1e-7)  # 精度要改吗？


def noise(mean_value, variance):
    return np.random.normal(loc=mean_value, scale=variance, size=None)


def test_2_twohills(solution):
    f1 = -math.pow(0.4 * solution[0] - 5, 2) - 2 * math.pow(0.4 * solution[1] - 17, 2) + 7
    f2 = -math.pow(0.4 * solution[0] - 12, 2) - math.pow(0.4 * solution[1] - 4, 2) + 4
    res = max(f1, f2)
    res = max(res, 0)
    return res + noise(0, 50)


def test_2_twohills_withoutnoise(solution):
    f1 = -math.pow(0.4 * solution[0] - 5, 2) - 2 * math.pow(0.4 * solution[1] - 17, 2) + 7
    f2 = -math.pow(0.4 * solution[0] - 12, 2) - math.pow(0.4 * solution[1] - 4, 2) + 4
    res = max(f1, f2)
    res = max(res, 0)
    return res


def h_q(q):
    w = (q / 60) / 1
    return 1570 * pow(w, 0.55) * (1 - 0.00075 * 30) / 9


def function(X,model):
    h = []
    h.append(h_q(X[0]))
    h.append(h_q(X[1]))
    h.append(h_q(X[2]))
    h.append(h_q(X[3]))
    # print(X)
    add_noise = 0
    # return  lossFunction(h,0.53,1530,0,model)-0.4        # 加速
    # return lossFunction(h, 0.53, 1530, 1, model) - 0.4,add_noise     # 不加速
    add_noise = max(0.4, min(0.6, 0.53 + noise(0, 0.1)))
    add_noise = round(add_noise, 2)
    return lossFunction(h, add_noise, 1530, 1, model), add_noise  # 不加速，加噪音
    # return h[0]+h[1]+h[2]+h[3]
    # add_noise = max(0.4, min(0.6, 0.53 + noise(0, 0.1)))
    # # add_noise = max(0.4, min(0.6, 0.6 + noise(0, 0.1)))
    # add_noise = round(add_noise, 2)
    # # print("add_noise", add_noise)
    # return lossFunction(h, add_noise, 1530) - 0.4

    # return  GetResultMLP(h, 0.427,1530)-0.4
    # return  GetResultMLP(h, 0.42721678053214446,1530)-0.4
    # return GetResult(h,0.6,1530)               #+0.5
    # return (pow(math.sin(math.sqrt(pow(X[0], 2)+pow(X[1], 2))),2)-0.5)/pow(1+0.001*(pow(X[0], 2)+pow(X[1], 2)),2)


main()