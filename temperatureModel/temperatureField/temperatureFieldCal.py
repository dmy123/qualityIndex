from temperatureModel.temperatureField import temperature_cal
import torch
import time
import scipy.io
import xlrd
import numpy as np
from pakage import continuousCastVariable

# 封装求解温度场程序，及提供两种形式温度场
# 根据输入数据集得到温度场数据集
# 封装求解温度场程序，及提供一种形式温度场

def one_example_temp_cal(ccv):
    MiddleTemp_all = temperature_cal.steady_temp_cal(ccv.var_dis, ccv.var_VcastOriginal, ccv.var_deltTime, ccv.MiddleTemp,
                                                        ccv.var_XNumber, ccv.var_YNumber, ccv.var_X, ccv.var_Y,
                                                        ccv.var_temperatureWater, ccv.var_rouS, ccv.var_rouL, ccv.var_specificHeatS,
                                                        ccv.var_specificHeatL, ccv.var_TconductivityS,
                                                        ccv.var_TconductivityL, ccv.var_liqTemp, ccv.var_SodTemp, ccv.var_m,
                                                        ccv.var_latentHeatofSolidification, ccv.Time_all,
                                                        ccv.time_Mold, ccv.var_h_initial, ccv.var_ZNumber, ccv.var_castingTemp)
    start_time = time.time()
    t = np.array(MiddleTemp_all).transpose(2, 1, 0).swapaxes(1, 2).tolist()
    # t = []
    # for i in range(ccv.Time_all):
    #     t.append([[0.0 for i in range(ccv.var_XNumber)] for j in range(ccv.var_YNumber)])
    # for stepTime in range(ccv.Time_all):
    #     for i in range(ccv.var_XNumber):
    #         for j in range(ccv.var_YNumber):
    #             t[stepTime][i][j] = MiddleTemp_all[i][j][stepTime]
    end_time = time.time()
    cal_time = end_time - start_time
    # print('得到t的时间：', cal_time)
    return MiddleTemp_all, t

# def one_example_temp_cal_input(ccv):
#     MiddleTemp_all = temperature_cal.steady_temp_cal(ccv.var_dis, ccv.var_VcastOriginal, ccv.var_deltTime, ccv.MiddleTemp,
#                                                         ccv.var_XNumber, ccv.var_YNumber, ccv.var_X, ccv.var_Y,
#                                                         ccv.var_temperatureWater, ccv.var_rouS, ccv.var_rouL, ccv.var_specificHeatS,
#                                                         ccv.var_specificHeatL, ccv.var_TconductivityS,
#                                                         ccv.var_TconductivityL, ccv.var_liqTemp, ccv.var_SodTemp, ccv.var_m,
#                                                         ccv.var_latentHeatofSolidification, ccv.Time_all,
#                                                         ccv.time_Mold, ccv.var_h_initial, ccv.var_ZNumber, ccv.var_castingTemp)
#     return MiddleTemp_all


def tempfieldcal():
    data = xlrd.open_workbook('/tmp/pycharm_project_82/temperatureModel/temperatureField/h_v_t.xlsx')
    sheet = data.sheet_by_name('Sheet1')
    all_rows = sheet.get_rows()
    # temperature_field_data = torch.empty((240,32,32,1920))
    temperature_field_data = torch.empty((240, 16,16, 96))
    count = 0
    cost_time = 0
    for row in all_rows:
        var_h_initial = [row[0].value,row[1].value,row[2].value,row[3].value]
        v_cast = row[4].value
        t_cast = row[5].value
        print("第",count,"次运行，参数：",var_h_initial,v_cast,t_cast)
        ccv = continuousCastVariable.ContinuousCastVariable(var_h_initial,v_cast,t_cast)
        start_time = time.time()
        MiddleTemp_all, t = one_example_temp_cal(ccv)
        end_time = time.time()
        cal_time = end_time - start_time
        print("计算一次温度场时间：", cal_time)
        cost_time = cost_time+cal_time
        temperature_field_data[count] = torch.Tensor(np.array(MiddleTemp_all)[::2,::2,::20])
        count  = count + 1
    scipy.io.savemat('../data/temperature_data.mat', mdict={'temperature_field_data':temperature_field_data.cpu().numpy()})

def onetempfieldcal(ccv):
    start_time = time.time()
    MiddleTemp_all = temperature_cal.steady_temp_cal(ccv.var_dis, ccv.var_VcastOriginal, ccv.var_deltTime,
                                                        ccv.MiddleTemp,
                                                        ccv.var_XNumber, ccv.var_YNumber, ccv.var_X, ccv.var_Y,
                                                        ccv.var_temperatureWater, ccv.var_rouS, ccv.var_rouL,
                                                        ccv.var_specificHeatS,
                                                        ccv.var_specificHeatL, ccv.var_TconductivityS,
                                                        ccv.var_TconductivityL, ccv.var_liqTemp, ccv.var_SodTemp,
                                                        ccv.var_m,
                                                        ccv.var_latentHeatofSolidification, ccv.Time_all_a,
                                                        ccv.time_Mold, ccv.var_h_initial, ccv.var_ZNumber,
                                                        ccv.var_castingTemp)
    end_time = time.time()
    cal_time = end_time - start_time
    # print("计算输入温度场时间：", cal_time)
    # print(MiddleTemp_all.shape)
    # preloc = np.zeros((ccv.var_XNumber, ccv.var_YNumber, ccv.Time_all - ccv.Time_all_a))
    # MiddleTemp_all = np.concatenate((MiddleTemp_all, preloc), axis=2)
    # print(MiddleTemp_all.shape)
    # temperature_field_data[0] = torch.Tensor(MiddleTemp_all)
    # scipy.io.savemat('oneInstance.mat', mdict={'temperature_field_data':temperature_field_data.cpu().numpy()})
    # scipy.io.savemat('oneInstance.mat',
    #                  mdict={'temperature_field_data': temperature_field_data.cpu().numpy()})
    # scipy.io.savemat('oneInstance.mat',
    #                  mdict={'temperature_field_data': MiddleTemp_all})
    # print(type(temperature_field_data.cpu().numpy()))
    # print(temperature_field_data.cpu().numpy().shape)
    return MiddleTemp_all


#生产训练测试温度场数据集
# tempfieldcal()