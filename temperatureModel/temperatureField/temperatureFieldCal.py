from temperatureModel.temperatureField import temperature_cal
import torch
import time
import scipy.io
import xlrd
import numpy as np
from pakage import continuousCastVariable


def one_example_temp_cal(ccv):
    MiddleTemp_all, t = temperature_cal.steady_temp_cal(ccv.var_dis, ccv.var_VcastOriginal, ccv.var_deltTime, ccv.MiddleTemp,
                                                        ccv.var_XNumber, ccv.var_YNumber, ccv.var_X, ccv.var_Y,
                                                        ccv.var_temperatureWater, ccv.var_rouS, ccv.var_rouL, ccv.var_specificHeatS,
                                                        ccv.var_specificHeatL, ccv.var_TconductivityS,
                                                        ccv.var_TconductivityL, ccv.var_liqTemp, ccv.var_SodTemp, ccv.var_m,
                                                        ccv.var_latentHeatofSolidification, ccv.Time_all,
                                                        ccv.time_Mold, ccv.var_h_initial, ccv.var_ZNumber, ccv.var_castingTemp)
    return MiddleTemp_all, t


def tempfieldcal():
    data = xlrd.open_workbook('/tmp/pycharm_project_82/temperatureModel/temperatureField/h_v_t.xlsx')
    sheet = data.sheet_by_name('Sheet1')
    all_rows = sheet.get_rows()
    temperature_field_data = torch.empty((240,32,32,1920))
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
        temperature_field_data[count] = torch.Tensor(MiddleTemp_all)
        count  = count + 1
    scipy.io.savemat('../data/temperature_data.mat', mdict={'temperature_field_data':temperature_field_data.cpu().numpy()})

def onetempfieldcal(ccv):
    start_time = time.time()
    MiddleTemp_all, t = temperature_cal.steady_temp_cal(ccv.var_dis, ccv.var_VcastOriginal, ccv.var_deltTime,
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
    print("计算输入温度场时间：", cal_time)
    MiddleTemp_all = np.array(MiddleTemp_all)
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