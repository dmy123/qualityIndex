import numpy as np
import math

class ContinuousCastVariable(object):
    def __init__(self,var_h_initial,v_cast,t_cast):
        # var_h_initial=[112.5,113.167,61.63,52.36]
        # var_h_initial=[163.65,134.36,68.07,46.47]
        self.var_ZNumber = 500  # 拉坯方向网格划分的数量
        # var_XNumber=20 # 铸坯厚度方向的网格点数
        self.var_XNumber = 32
        self.var_X = 0.16  # 铸坯厚度
        # var_YNumber=20 # 铸坯厚度方向的网格点数
        self.var_YNumber = 32
        self.var_Y = 0.16  # 铸坯厚度
        self.var_Z = 7.68
        self.deltZ = self.var_Z / self.var_ZNumber  # 拉皮方向的空间间隔
        # self.middle_temp = [([t_cast] * self.var_XNumber) for i in range(self.var_XNumber)]
        self.var_temperatureWater = 30
        self.var_rouS = 7800
        self.var_rouL = 7200
        self.var_specificHeatS = 660
        self.var_specificHeatL = 830
        self.var_TconductivityS = 31
        self.var_TconductivityL = 35
        self.var_liqTemp = 1514
        self.var_SodTemp = 1475
        self.var_m = 7.1
        self.var_latentHeatofSolidification = 268000
        self.var_controlTime = 3
        self.var_dis = [0.0, 0.9, 1.27, 3.12, 5.32, 7.68]  # 连铸二冷区各段距弯月面的距离 单位m
        self.var_runningTime = 20
        self.var_VcastOriginal = v_cast / 60
        self.var_castingTemp = t_cast  # 铸坯的浇筑温度
        self.var_deltTime = 0.4 * 0.6 / v_cast
        self.MiddleTemp = [([self.var_castingTemp] * self.var_YNumber) for i in range(self.var_XNumber)]
        self.tl = [0] * len(self.var_dis);  # 铸坯凝固时间的初值var_dis=[0.0,0.9,1.27,3.12,5.32,7.68] # 连铸二冷区各段距弯月面的距离 单位m
        self.t_l = np.ones(len(self.var_dis));
        for i in range(len(self.var_dis)):
            self.tl[i] = self.var_dis[i] / (v_cast / 60)  # var_VcastOriginal=0.6/60
            self.t_l[i] = int(self.tl[i] / self.var_deltTime) - 2
        self.time_Mold = int((self.tl[1] - self.tl[0]) / self.var_deltTime)
        self.time_SCZ = int((self.tl[len(self.var_dis) - 1] - self.tl[1]) / self.var_deltTime)  # var_deltTime=0.4 # 差分计算时间间隔
        # Time_all = time_Mold + time_SCZ
        self.Time_all = 1920
        self.Time_all_a = 960
        self.v_cast = v_cast
        self.t_cast = t_cast
        self.var_h_initial = var_h_initial

        # 具体数值需查询的参数 start
        # 坯壳厚度
        self.strand_shell_set = 0.02  # 0 < strand_shell_set < min(var_X, var_Y)/2
        self.strand_shell_set_loc = math.ceil(self.strand_shell_set * self.var_XNumber / self.var_X)  # 坯壳厚度对应坐标

        self.max_surface_temperature = 1100
        self.min_surface_temperature = 800
        # 温升
        self.t_up_lim = 150
        self.t_down_lim = 200
        self.temp_rise = 1
        self.D_T_uplim = self.t_up_lim * self.temp_rise
        self.D_T_downlim = self.t_down_lim * self.temp_rise

        # 矫直点温度
        # Tj_min = 900
        # Tj_max = 1300
        self.Tj_min = 900
        self.Tj_max = 1100

        # 具体数值需查询的参数 end
        self.straighten_point = int(self.t_l[-1] - 300)