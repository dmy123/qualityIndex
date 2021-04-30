import math
import time

def fs(Ts, T, Tl):  # 固相率
    if T < Ts:
        return 1
    elif T > Tl:
        return 0
    else:
        return (Tl - T) / (Tl - Ts)

def crack(MiddleTemp_all, var_XNumber, var_X, var_YNumber, var_Y, Time_all, var_liqTemp, var_SodTemp):
    # 定义收缩系数(系数待定)
    # var_YNumber=1920
    alpha_l = 5  # 液态收缩率
    beta = 2.5  # 凝固收缩率
    alpha_s = 1  # 固态收缩率
    fso = 0.65  # 定义中心固相率
    start_t = 0  # 不具备补缩能力开始的时刻
    nx = 1000 * var_X / var_XNumber
    ny = 1000 * var_Y / var_YNumber
    S_nxny = [[nx * ny] * var_YNumber for i in range(var_XNumber)]  # 小单元面积
    x_nxny = [[var_X / var_XNumber] * var_YNumber for i in range(var_XNumber)]  # 小单元x
    y_nxny = [[var_Y / var_YNumber] * var_YNumber for i in range(var_XNumber)]  # 小单元y
    det_S_xy = [0] * Time_all  # 每次总面积变化
    S_xy = [0] * Time_all
    # 找到不具备补缩能力的时间
    start_time = time.time()
    for t in range(Time_all):
        if fs(var_SodTemp, MiddleTemp_all[0][0][t], var_liqTemp) >= fso:
            start_t = t + 1
            break
    end_time = time.time()
    cal_time = end_time - start_time
    # print('找到不具备补缩能力的时间：', cal_time)
    # print(start_t)
    # 计算
    sum2 = 0  # 实际总和面积
    x = 0
    y = 0

    start_time = time.time()
    for t in range(start_t, Time_all):
        for i in range(var_XNumber):
            for j in range(var_YNumber):
                fsxy = fs(var_SodTemp, MiddleTemp_all[i][j][t], var_liqTemp)

                fsxy1 = fs(var_SodTemp, MiddleTemp_all[i][j][t - 1], var_liqTemp)

                # 单元收缩核心算法
                epsilon_l = alpha_l * (MiddleTemp_all[i][j][t] - MiddleTemp_all[i][j][t - 1])

                epsilon_ls = beta * (MiddleTemp_all[i][j][t] - MiddleTemp_all[i][j][t - 1])
                epsilon_s = alpha_s * (MiddleTemp_all[i][j][t] - MiddleTemp_all[i][j][t - 1])
                epsilon_evp = epsilon_l * (1 - fsxy) + epsilon_ls * (fsxy - fsxy1) + epsilon_s * fsxy1
                ex = (epsilon_evp) / 3

                x_nxny[i][j] = x_nxny[i][j] * (1 - abs(ex))
                y_nxny[i][j] = y_nxny[i][j] * (1 - abs(ex))
                sum2 += x_nxny[i][j] * y_nxny[i][j]  # 实际总和面积
                # print("x1y1",epsilon_ls,epsilon_s,epsilon_evp,(MiddleTemp_all[i][j][t]-MiddleTemp_all[i][j][t-1]))
        S_xy[t] = sum2
        # 计算边长总和
        for i in range(var_XNumber):
            x = x + x_nxny[i][0]
        for j in range(var_YNumber):
            y = y + x_nxny[0][j]
        det_S = 4 * (x * y - sum2)
        # print("xy",x,y)
        if det_S < 0:
            result = 0  ##有错误
        else:

            result = math.sqrt(det_S / 3.14)
        det_S_xy[t] = result
    # print("tu",det_S_xy)
    # print(S_xy)
    # print(x_nxny)
    end_time = time.time()
    cal_time = end_time - start_time
    # print('crack时间：', cal_time)
    return det_S_xy[Time_all - 1], det_S_xy