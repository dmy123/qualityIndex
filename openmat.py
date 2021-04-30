
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
#查看mat文件信息
# dataFile = '/tmp/pycharm_project_82/one_temperature_field_eval.mat'
# dataFile ='/tmp/pycharm_project_82/temperatureModel/temperatureField/oneInstance.mat'
# data = scio.loadmat(dataFile)
# print(data.keys())
# print(type(data))
# print(data['pred'][0][:][:][0])
# print(data['pred'].shape)
# print(data['u'][0][:][:][0])
# print(data['u'].shape)

# 查看数据集
# dataFile = 'temperatureModel/data/temperature_data.mat'
# dataFile = '/tmp/pycharm_project_82/temperatureModel/temperatureField/oneInstance.mat'
# data = scio.loadmat(dataFile)
# print(data['temperature_field_data'])
# print(data['temperature_field_data'].shape)
# for example in range(2,240,241):
#     pred_output = data['temperature_field_data'][example]
#     # for time in range(0,1920,500):
#     for time in range(0, 96, 24):
#         pred_output_slice = pred_output[:, :, time]
#         plt.imshow(pred_output_slice)
#         plt.axis('on')
#         plt.xlabel("example_"+str(example)+" predict_time_"+str(time))
#         plt.show()

#对比预测与实际温度场
dataFile = '/tmp/pycharm_project_82/temperatureModel/pred/temperature_field_eval.mat'
# dataFile = '/tmp/pycharm_project_82/one_temperature_field_eval.mat'
data = scio.loadmat(dataFile)
print(data.keys())
print(type(data))
#print(data['temperature_field_data'])
print('pred.shape:',data['pred'].shape,',u.shape:',data['u'].shape)
for example in range(1):
    pred_output = data['pred'][example]
    u_output = data['u'][example]
    for time in range(0, 96, 12):
        # pred_output_slice = (pred_output[:,:,time]).astype(np.int)
        pred_output_slice = pred_output[:, :, time]
        u_output_slice = u_output[:, :, time]
        plt.imshow(pred_output_slice)
        plt.axis('on')
        # plt.xlabel("example_" + str(example) + " predict_time_" + str(time))
        plt.show()
        plt.imshow(u_output_slice)
        plt.axis('on')
        # plt.xlabel("example_" + str(example) + " u_time_" + str(time))
        plt.show()

pred = data['pred']
test_u = data['u']
cost = 0
for time in range(960):
    for x in range(32):
        for y in range(32):
            cost = cost + abs(pred[0][x][y][time] -test_u[0][x][y][time])
print('花费：',cost/960)