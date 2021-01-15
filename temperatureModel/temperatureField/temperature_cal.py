import math

#计算下一时刻温度场
def two_densonal_diff(h, var_deltTime, middle_temp, var_XNumber, var_YNumber, var_X, var_Y, var_temperatureWater,
                      var_rouS, var_rouL, var_specificHeatS, var_specificHeatL, var_TconductivityS, var_TconductivityL,
                      var_liqTemp, var_SodTemp, var_m, var_latentHeatofSolidification):
    deltX = var_X / var_XNumber  # 铸坯在厚度方向的空间间隔
    deltY = var_Y / var_YNumber  # 铸坯在宽度方向的空间间隔
    BOLTZMAN = 0.000000056684  # 玻尔兹曼常数
    EMISSIVITY = 0.8  # 辐射系数
    next_temp = [[0] * var_YNumber for i in range(var_XNumber)]
    #### equation strat #####
    for i in range(var_XNumber):
        for j in range(var_YNumber):
            if middle_temp[i][j] >= var_liqTemp:
                rou = var_rouL
                specificHeat = var_specificHeatL
                Tconductivity = var_TconductivityL
            if middle_temp[i][j] <= var_SodTemp:
                rou = var_rouS
                specificHeat = var_specificHeatS
                Tconductivity = var_TconductivityS
            if (var_SodTemp < middle_temp[i][j]) & (var_liqTemp > middle_temp[i][j]):
                rou = (var_rouS - var_rouL) * (var_liqTemp -
                                               middle_temp[i][j]) / (var_liqTemp - var_SodTemp) + var_rouL
                Tconductivity = (var_TconductivityS) * (var_liqTemp - middle_temp[i][j]) / (
                            var_liqTemp - var_SodTemp) + var_m * (
                                        1 - (var_liqTemp - middle_temp[i][j]) / (
                                            var_liqTemp - var_SodTemp)) * var_TconductivityL
                specificHeat = (var_specificHeatS - var_specificHeatL) * (var_liqTemp - middle_temp[i][j]) / (
                        var_liqTemp - var_SodTemp) + var_specificHeatL + var_latentHeatofSolidification / (
                                           var_liqTemp - var_SodTemp)
            # a1 = Tconductivity/(rou*specificHeat)
            a = (Tconductivity * var_deltTime) / (rou * specificHeat * deltX * deltX)
            b = (Tconductivity * var_deltTime) / (rou * specificHeat * deltY * deltY)
            ##### 四个边的温度 start ######
            if i == 0 and (j != 0 and j != var_YNumber - 1):  # 情况1
                next_temp[i][j] = middle_temp[i][j] + 2 * a * (middle_temp[i + 1][j] - middle_temp[i][j]) + b * (
                            middle_temp[i][j + 1] - 2 * middle_temp[i][j] + middle_temp[i][j - 1])
            if j == 0 and (i != var_XNumber - 1 and i != 0):  # 情况2
                next_temp[i][j] = middle_temp[i][j] + 2 * a * (middle_temp[i][j + 1] - middle_temp[i][j]) + b * (
                            middle_temp[i + 1][j] - 2 * middle_temp[i][j] + middle_temp[i - 1][j])
            if i == var_XNumber - 1 and (j != var_YNumber - 1 and j != 0):  # 情况3
                h_int = h + EMISSIVITY * BOLTZMAN * (
                            middle_temp[i][j] * middle_temp[i][j] + var_temperatureWater * var_temperatureWater) * (
                                    middle_temp[i][j] + var_temperatureWater)
                next_temp[i][j] = middle_temp[i][j] + 2 * a * (middle_temp[i - 1][j] - middle_temp[i][j]) + b * (
                            middle_temp[i][j + 1] - 2 * middle_temp[i][j] + middle_temp[i][j - 1]) - (
                                              2 * h_int * deltX * a * (
                                                  middle_temp[i][j] - var_temperatureWater) / Tconductivity)
            if j == var_YNumber - 1 and (i != var_XNumber - 1 and i != 0):  # 情况4
                h_int = h + EMISSIVITY * BOLTZMAN * (
                            middle_temp[i][j] * middle_temp[i][j] + var_temperatureWater * var_temperatureWater) * (
                                    middle_temp[i][j] + var_temperatureWater)
                next_temp[i][j] = middle_temp[i][j] + 2 * b * (middle_temp[i][j - 1] - middle_temp[i][j]) + a * (
                            middle_temp[i + 1][j] - 2 * middle_temp[i][j] + middle_temp[i - 1][j]) - (
                                              2 * h_int * deltY * b * (
                                                  middle_temp[i][j] - var_temperatureWater) / Tconductivity)
            ##### 四个边的温度 end ######
            ##### 四个角的温度 start ####
            if i == 0 and j == 0:  # 情况5
                next_temp[i][j] = middle_temp[i][j] + 2 * a * (middle_temp[i + 1][j] - middle_temp[i][j]) + 2 * b * (
                            middle_temp[i][j + 1] - middle_temp[i][j])
            if i == 0 and j == var_YNumber - 1:  # 情况6
                h_int = h + EMISSIVITY * BOLTZMAN * (
                            middle_temp[i][j] * middle_temp[i][j] + var_temperatureWater * var_temperatureWater) * (
                                    middle_temp[i][j] + var_temperatureWater)
                next_temp[i][j] = middle_temp[i][j] + 2 * a * (middle_temp[i + 1][j] - middle_temp[i][j]) + 2 * b * (
                            middle_temp[i][j - 1] - middle_temp[i][j]) - (2 * h_int * deltY * b * (
                            middle_temp[i][j] - var_temperatureWater) / Tconductivity)
            if i == var_XNumber - 1 and j == 0:  # 情况7
                h_int = h + EMISSIVITY * BOLTZMAN * (
                            middle_temp[i][j] * middle_temp[i][j] + var_temperatureWater * var_temperatureWater) * (
                                    middle_temp[i][j] + var_temperatureWater)
                next_temp[i][j] = middle_temp[i][j] + 2 * a * (middle_temp[i - 1][j] - middle_temp[i][j]) + 2 * b * (
                            middle_temp[i][j + 1] - middle_temp[i][j]) - (2 * h_int * deltX * a * (
                            middle_temp[i][j] - var_temperatureWater) / Tconductivity)
            if i == var_XNumber - 1 and j == var_YNumber - 1:  # 情况8
                h_int = h + EMISSIVITY * BOLTZMAN * (
                            middle_temp[i][j] * middle_temp[i][j] + var_temperatureWater * var_temperatureWater) * (
                                    middle_temp[i][j] + var_temperatureWater)
                next_temp[i][j] = middle_temp[i][j] + 2 * a * (middle_temp[i - 1][j] - middle_temp[i][j]) + 2 * b * (
                            middle_temp[i][j - 1] - middle_temp[i][j]) - (2 * h_int * deltY * b * (
                            middle_temp[i][j] - var_temperatureWater) / Tconductivity) - (2 * h_int * deltX * a * (
                            middle_temp[i][j] - var_temperatureWater) / Tconductivity)
            ##### 四个角的温度 end ####
            ##### 内部温度 start ####
            if (i != 0 and i != var_XNumber - 1) and (j != 0 and j != var_YNumber - 1):  # 情况9
                next_temp[i][j] = middle_temp[i][j] + a * (
                            middle_temp[i + 1][j] - 2 * middle_temp[i][j] + middle_temp[i - 1][j]) + b * (
                                              middle_temp[i][j + 1] - 2 * middle_temp[i][j] + middle_temp[i][j - 1])
            ##### 内部温度 end ####
    return next_temp

#计算整个温度场
def steady_temp_cal(var_dis, var_VcastOriginal, var_deltTime, MiddleTemp, var_XNumber, var_YNumber, var_X, var_Y,
                    var_temperatureWater, var_rouS, var_rouL, var_specificHeatS, var_specificHeatL, var_TconductivityS,
                    var_TconductivityL, var_liqTemp, var_SodTemp, var_m, var_latentHeatofSolidification, Time_all,
                    time_Mold, var_h_initial, var_sliceNumber, var_castingTemp):
    NextTemp = [([0] * var_YNumber) for i in range(var_XNumber)]
    MiddleTemp_all = [0] * var_XNumber
    for i in range(var_XNumber):
        MiddleTemp_all[i] = [0] * var_YNumber
    for i in range(var_XNumber):
        for j in range(var_YNumber):
            MiddleTemp_all[i][j] = [0] * Time_all
    t = []
    for i in range(Time_all):
        t.append([[0 for i in range(var_XNumber)] for j in range(var_YNumber)])

    for stepTime in range(Time_all):
        if stepTime <= time_Mold:
            tTime = var_deltTime * (stepTime + 1);
            h = 1000 * (0.07128 * math.exp(-tTime) + 2.328 * math.exp(-tTime / 9.5) + 0.698)
            NextTemp = two_densonal_diff(h, var_deltTime, MiddleTemp, var_XNumber, var_YNumber, var_X, var_Y,
                                         var_temperatureWater, var_rouS, var_rouL, var_specificHeatS, var_specificHeatL,
                                         var_TconductivityS, var_TconductivityL, var_liqTemp, var_SodTemp, var_m,
                                         var_latentHeatofSolidification)
        else:
            disNow = var_dis[0] + stepTime * var_VcastOriginal * var_deltTime
            if var_dis[1] <= disNow <= var_dis[2]:
                h = var_h_initial[0]
            if var_dis[2] < disNow <= var_dis[3]:
                h = var_h_initial[1]
            if var_dis[3] < disNow <= var_dis[4]:
                h = var_h_initial[2]
            if var_dis[4] < disNow <= var_dis[5]:
                h = var_h_initial[3]
            NextTemp = two_densonal_diff(h, var_deltTime, MiddleTemp, var_XNumber, var_YNumber, var_X, var_Y,
                                         var_temperatureWater, var_rouS, var_rouL, var_specificHeatS, var_specificHeatL,
                                         var_TconductivityS, var_TconductivityL, var_liqTemp, var_SodTemp, var_m,
                                         var_latentHeatofSolidification)
        for i in range(var_XNumber):
            for j in range(var_YNumber):
                MiddleTemp[i][j] = NextTemp[i][j]
                MiddleTemp_all[i][j][stepTime] = NextTemp[i][j]
                t[stepTime][i][j] = NextTemp[i][j]
    return MiddleTemp_all, t

#冶金准则
def constrain_punish(t, tl, t_l, time_Mold, var_XNumber, var_YNumber, var_SodTemp, strand_shell_set_loc,
                     max_surface_temperature, min_surface_temperature, temp_rise, D_T_uplim, D_T_downlim,
                     straighten_point, Tj_min, Tj_max):
    y_index = t.shape[1]
    x_index = t.shape[2]
    # print(y_index,x_index)
    punish = 0
    strand_shell_punish = 0
    surface_temp = 0
    liquid_core = 0
    temp_rise_down = 0
    straiten_temp = 0
    punish_list = []
    for find_shell in range(x_index - 1, x_index - 1 - strand_shell_set_loc, -1):
        y_loc = int(y_index / 2)
        # print(t[time_Mold][y_loc][find_shell])
        if (t[time_Mold][y_loc][find_shell] > var_SodTemp):
            # print(t[time_Mold][y_loc][find_shell])
            strand_shell_punish = strand_shell_set_loc + find_shell - x_index + 1
            break
    # print("坯壳安全厚度惩罚",strand_shell_punish)
    punish_list.append(strand_shell_punish)
    punish += strand_shell_punish / strand_shell_set_loc  # 距离坯壳安全厚度几个单位距离 0

    coolzone = 1  # 三个区
    max_temp = max_surface_temperature
    min_temp = min_surface_temperature
    for i in range(0, (t.shape[0] - 2)):
        if (i == t_l[coolzone]):
            coolzone = coolzone + 1
            # print(coolzone)
            surface_temp += ((max_temp - max_surface_temperature) / max_surface_temperature + (
                        min_surface_temperature - min_temp) / min_surface_temperature)

            max_temp = max_surface_temperature
            min_temp = min_surface_temperature
            # print("max_temp",max_temp,min_temp,surface_temp)
        for j in range(y_index):
            max_temp = max(max_temp, t[i][j][x_index - 1])
            min_temp = min(min_temp, t[i][j][x_index - 1])
        for k in range(x_index - 1):
            max_temp = max(max_temp, t[i][y_index - 1][k])
            min_temp = min(min_temp, t[i][y_index - 1][k])
        # print("max_temp",max_temp,min_temp)

    surface_temp += ((max_temp - max_surface_temperature) / max_surface_temperature + (
                min_surface_temperature - min_temp) / min_surface_temperature)
    surface_temp = surface_temp / 3
    # surface_temp=surface_temp/10
    # print("表面温度惩罚",surface_temp)
    punish_list.append(surface_temp)
    punish += surface_temp  # 表面温度超出最大最小温度的值2612907-2553
    # print("straighten_point",straighten_point)
    for i in range(straighten_point, t.shape[0]):
        if (t[i][0][0] < var_SodTemp or i == t.shape[0] - 1):
            liquid_core = (i - straighten_point) / straighten_point
            break
    # print("液芯长度限制",liquid_core)
    punish_list.append(liquid_core)
    punish += liquid_core  # 液芯长度限制2612907-2612632
    # t_l
    for j in range(y_index):
        for k in range(x_index):
            rise_temp1 = max((t[int(t_l[2])][j][k] - t[int(t_l[1])][j][k]) * (tl[2] - tl[1]), D_T_uplim)
            down_temp1 = max((t[int(t_l[1])][j][k] - t[int(t_l[2])][j][k]) * (tl[2] - tl[1]), D_T_downlim)
            rise_temp2 = max((t[int(t_l[3])][j][k] - t[int(t_l[2])][j][k]) * (tl[3] - tl[2]), D_T_uplim)
            down_temp2 = max((t[int(t_l[2])][j][k] - t[int(t_l[3])][j][k]) * (tl[3] - tl[2]), D_T_downlim)
            rise_temp3 = max((t[int(t_l[4])][j][k] - t[int(t_l[3])][j][k]) * (tl[4] - tl[3]), D_T_uplim)
            down_temp3 = max((t[int(t_l[3])][j][k] - t[int(t_l[4])][j][k]) * (tl[4] - tl[3]), D_T_downlim)
            rise_temp4 = max((t[int(t_l[5])][j][k] - t[int(t_l[4])][j][k]) * (tl[5] - tl[4]), D_T_uplim)
            down_temp4 = max((t[int(t_l[4])][j][k] - t[int(t_l[5])][j][k]) * (tl[5] - tl[4]), D_T_downlim)

    temp_rise_down = ((
                                  rise_temp1 + down_temp1 + rise_temp2 + down_temp2 + rise_temp3 + down_temp3 + rise_temp4 + down_temp4) - 4 * (
                                  D_T_uplim + D_T_downlim)) / 200

    # print("温升温降限制",temp_rise_down)
    punish_list.append(temp_rise_down)
    punish += temp_rise_down  # 0

    straighten_point = int(t_l[-1])  # 矫直点
    t_straiten_max = Tj_max
    t_straiten_min = Tj_min
    t_straiten_max = t[straighten_point][0][x_index - 1]
    t_straiten_min = t[straighten_point][y_index - 1][x_index - 1]

    if (t_straiten_max > Tj_max):
        straiten_temp += (t_straiten_max - Tj_max) / Tj_max

    if (t_straiten_max < Tj_min):
        straiten_temp += (Tj_min - t_straiten_min) / Tj_min
    # print("矫直点表面温度限制",straiten_temp)
    punish_list.append(straiten_temp)
    punish += straiten_temp  # 矫直点表面温度限制2612907-2610628
    punish_list.append(punish)
    return punish_list
