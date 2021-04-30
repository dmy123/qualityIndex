from crackAndStress.stress import solver
from crackAndStress.crack import crack
from temperatureModel.temperatureField.temperatureFieldCal import one_example_temp_cal
from temperatureModel.temperatureField.temperature_cal import constrain_punish
from GetOneTemperatureField import getOneTempField
from pakage import continuousCastVariable
import numpy as np
import matplotlib.pyplot as plt
import time

def paintcrack_stress(var_h_initial, v_cast, t_cast):
    ccv = continuousCastVariable.ContinuousCastVariable(var_h_initial, v_cast, t_cast)
    MiddleTemp_all22, t = one_example_temp_cal(ccv)
    for cattime in range(ccv.Time_all,400):
        slice = MiddleTemp_all22[:][:][cattime]
        plt.plot(slice)
        plt.imshow(slice)
        plt.axis('on')
        # plt.xlabel("example_" + str(example) + " u_time_" + str(time))
        plt.show()
    surfacetemp = MiddleTemp_all22[ccv.var_YNumber-1][ccv.var_XNumber-1]
    result_crack, crack_list = crack(MiddleTemp_all22, ccv.var_XNumber, ccv.var_X, ccv.var_YNumber, ccv.var_Y,
                                     ccv.Time_all, ccv.var_liqTemp,
                                     ccv.var_SodTemp)  # 缩孔

    stress, estress, strain, f2 = solver(ccv.var_XNumber, ccv.var_YNumber, ccv.var_X / ccv.var_XNumber,
                                         ccv.var_Y / ccv.var_YNumber, t, int(ccv.t_l[-3]), int(ccv.t_l[-1]))  # 计算节点位移
    stress, estress1, strain, f3 = solver(ccv.var_XNumber, ccv.var_YNumber, ccv.var_X / ccv.var_XNumber,
                                          ccv.var_Y / ccv.var_YNumber, t, int(ccv.t_l[-2]), int(ccv.t_l[-1]))  # 计算节点位移
    stress, estress1, strain, f4 = solver(ccv.var_XNumber, ccv.var_YNumber, ccv.var_X / ccv.var_XNumber,
                                          ccv.var_Y / ccv.var_YNumber, t, int(ccv.t_l[-1]), int(ccv.t_l[-1]))  # 计算节点位移
    stress, estress1, strain, f1 = solver(ccv.var_XNumber, ccv.var_YNumber, ccv.var_X / ccv.var_XNumber,
                                          ccv.var_Y / ccv.var_YNumber, t, int(ccv.t_l[-4]), int(ccv.t_l[-1]))  # 计算节点位移
    mental_punish = constrain_punish(np.array(t), ccv.var_dis, ccv.t_l, ccv.time_Mold, ccv.var_XNumber, ccv.var_YNumber,
                                     ccv.var_SodTemp,
                                     ccv.strand_shell_set_loc, ccv.max_surface_temperature, ccv.min_surface_temperature,
                                     ccv.temp_rise,
                                     ccv.D_T_uplim, ccv.D_T_downlim, ccv.straighten_point, ccv.Tj_min, ccv.Tj_max)
    # return f1 / 7, result_crack, f2, f3, mental_punish, t, crack_list, strain, estress, f4 / 7
    return crack_list,surfacetemp

def comparecrack(h1,h2,v_cast, t_cast):
    crack_list1,surfacetemp1 = paintcrack_stress(h1,v_cast,t_cast)
    crack_list2,surfacetemp2 = paintcrack_stress(h2,v_cast,t_cast)
    length = len(crack_list1)
    print(crack_list1[length-1],crack_list2[length-1])
    plt.plot(np.arange(length), crack_list1,label='optimizated')
    plt.plot(np.arange(len(crack_list2)), crack_list2,linestyle='--',label='not optimizated')
    plt.xlabel('time slice')
    plt.ylabel('shrinkage cavity')
    plt.legend()
    plt.show()
    plt.plot(np.arange(len(surfacetemp1)), surfacetemp1,label='optimizated')
    plt.plot(np.arange(len(surfacetemp2)), surfacetemp2, linestyle='--',label='not optimizated')
    plt.xlabel('time slice')
    plt.ylabel('temperature')
    plt.legend()
    plt.show()

h1 = [  8.197538,36.99038841,20.13660549,12.63770882]
h2= [  25.5,28.5,9.15,6.8]
comparecrack(h1,h2,0.53,1530)