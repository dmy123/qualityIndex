import numpy as np
import math


def lnd_xy_creack(nx, ny, dx, dy):  # 创建节点编码矩阵nx,ny为边界的节点个数dx为单元边长
    unit = (nx - 1) * (ny - 1) * 2  # 单元个数
    lnd = [[0 for i in range(3)] for j in range(unit)]  # 每个单元的角标
    l = 0
    # '对各个单元编号,与节点编号对应 i=lnd(ie)(0)
    for j in range(ny - 1):
        for i in range(nx - 1):
            lnd[l][0] = i + j * (nx)
            lnd[l][1] = i + 1 + j * (nx)
            lnd[l][2] = i + (j + 1) * (nx)
            lnd[l + 1][0] = i + j * (nx) + 1
            lnd[l + 1][1] = i + 1 + (j + 1) * (nx)
            lnd[l + 1][2] = i + (j + 1) * (nx)
            l = l + 2
    m = nx * ny  # 总节点数
    xy = [[0 for i in range(4)] for j in range(m)]

    for i in range(ny):
        l = i * nx
        for j in range(nx):
            xy[l + j][0] = j * dx
            xy[l + j][1] = i * dy
            xy[l + j][2] = j  # x轴坐标
            xy[l + j][3] = i  # y轴坐标
    return lnd, xy, unit, m


# 几个计算过程用到的变量
def Ee(t):
    if t > 900:
        e = 31.46 - 0.02256 * t + 0.00000138 * pow(t, 2)
    else:
        e = 347.6526 - 0.3727 * t
    # e=e * 1000000000.0
    return e


def u(t):
    u = 0.287 + 0.0000823 * t
    return u


def a(t):
    aa = (0.435 + 0.000708 * t) * 0.00001
    return aa


def area(ie, lnd, xy):
    i = lnd[ie][0]  # i.j.k代表节点号
    j = lnd[ie][1]
    k = lnd[ie][2]
    area = (xy[j][0] - xy[i][0]) * (xy[k][1] - xy[i][1]) - (xy[k][0] - xy[i][0]) * (xy[j][1] - xy[i][1])
    area = area * 0.5
    return area


def betaf(ie, t):  # 这部分有疑问待解决
    beta = (1 - 2 * u(t)) / (1 + u(t))
    return beta


# 计算弹朔性矩阵
def md(ie, t):  # 这里的t是温度单个单元的温度
    beta = betaf(ie, t)
    d = [[0 for i in range(3)] for j in range(3)]
    d[0][0] = 2 + beta
    d[0][1] = 1 - beta
    d[1][0] = d[0][1]
    d[1][1] = d[0][0]
    d[2][2] = 0.5 + beta
    for i in range(3):
        for j in range(3):
            d[i][j] = beta * Ee(t) / (1 - 2 * u(t)) / (1 + 2 * beta) * d[i][j]

    return d


# 计算位移——应变转换矩阵（几何矩阵B）
def mb(ie, lnd, xy):  # ie代表的是单元序号
    i = lnd[ie][0]  # i.j.k代表节点号
    j = lnd[ie][1]
    k = lnd[ie][2]
    b = [[0 for i in range(6)] for j in range(3)]
    b[0][0] = xy[j][1] - xy[k][1]
    b[0][2] = xy[k][1] - xy[i][1]
    b[0][4] = xy[i][1] - xy[j][1]
    b[1][1] = xy[k][0] - xy[j][0]
    b[1][3] = xy[i][0] - xy[k][0]
    b[1][5] = xy[j][0] - xy[i][0]
    b[2][0] = b[1][1]
    b[2][1] = b[0][0]
    b[2][2] = b[1][3]
    b[2][3] = b[0][2]
    b[2][4] = b[1][5]
    b[2][5] = b[0][4]
    s = area(ie, lnd, xy)
    # s=0.5
    for ii in range(3):
        for jj in range(6):
            b[ii][jj] = b[ii][jj] / (2 * s)
    return b


# 求解单元刚度矩阵
def mke(ie, t, lnd, xy, kk):  # kk代表当前时间切片,这里的t是温度场
    i = lnd[ie][0]  # i.j.k代表节点号
    j = lnd[ie][1]
    k = lnd[ie][2]
    tpkk = t[kk]
    te = tpkk[xy[i][2]][xy[i][3]]
    te += tpkk[xy[j][2]][xy[j][3]]
    te += tpkk[xy[k][2]][xy[k][3]]
    te = te / 3.0
    d = md(ie, te)
    b = mb(ie, lnd, xy)  # 3*6的矩阵
    s = [[0 for i in range(6)] for j in range(3)]

    for ii in range(3):
        for jj in range(6):
            s[ii][jj] = 0
            for kk in range(3):
                s[ii][jj] = s[ii][jj] + d[ii][kk] * b[kk][jj]
    ake = [[0 for i in range(6)] for j in range(6)]
    for ii in range(6):
        for jj in range(6):
            ake[ii][jj] = 0
            for kk in range(3):
                ake[ii][jj] = ake[ii][jj] + s[kk][ii] * b[kk][jj]
    return ake


# 求解整体刚度矩阵
def mk(unit, m, t, lnd, xy, kk):
    N = 2 * m
    ak = [[0 for i in range(N)] for j in range(N)]
    for ie in range(unit):
        ake = mke(ie, t, lnd, xy, kk)
        i = lnd[ie][0]  # i.j.k代表节点号
        j = lnd[ie][1]
        k = lnd[ie][2]
        # print(i,j,k)
        ak[2 * i][2 * i] = ak[2 * i][2 * i] + ake[0][0]
        ak[2 * i][2 * i + 1] += ake[0][1]
        ak[2 * i][2 * j] += ake[0][2]
        ak[2 * i][2 * j + 1] += ake[0][3]
        ak[2 * i][2 * k] += ake[0][4]
        ak[2 * i][2 * k + 1] += ake[0][5]
        ak[2 * i + 1][2 * i] += ake[1][0]
        ak[2 * i + 1][2 * i + 1] += ake[1][1]
        ak[2 * i + 1][2 * j] += ake[1][2]
        ak[2 * i + 1][2 * j + 1] += ake[1][3]
        ak[2 * i + 1][2 * k] += ake[1][4]
        ak[2 * i + 1][2 * k + 1] += ake[1][5]

        ak[2 * j][2 * i] += ake[2][0]
        ak[2 * j][2 * i + 1] += ake[2][1]
        ak[2 * j][2 * j] += ake[2][2]
        ak[2 * j][2 * j + 1] += ake[2][3]
        ak[2 * j][2 * k] += ake[2][4]
        ak[2 * j][2 * k + 1] += ake[2][5]
        ak[2 * j + 1][2 * i] += ake[3][0]
        ak[2 * j + 1][2 * i + 1] += ake[3][1]
        ak[2 * j + 1][2 * j] += ake[3][2]
        ak[2 * j + 1][2 * j + 1] += ake[3][3]
        ak[2 * j + 1][2 * k] += ake[3][4]
        ak[2 * j + 1][2 * k + 1] += ake[3][5]

        ak[2 * k][2 * i] += ake[4][0]
        ak[2 * k][2 * i + 1] += ake[4][1]
        ak[2 * k][2 * j] += ake[4][2]
        ak[2 * k][2 * j + 1] += ake[4][3]
        ak[2 * k][2 * k] += ake[4][4]
        ak[2 * k][2 * k + 1] += ake[4][5]
        ak[2 * k + 1][2 * i] += ake[5][0]
        ak[2 * k + 1][2 * i + 1] += ake[5][1]
        ak[2 * k + 1][2 * j] += ake[5][2]
        ak[2 * k + 1][2 * j + 1] += ake[5][3]
        ak[2 * k + 1][2 * k] += ake[5][4]
        ak[2 * k + 1][2 * k + 1] += ake[5][5]
    return ak


# 计算热荷载
def heatload(t, kk, lnd, xy, unit, kk1, m):  # t,kk为温度场和时间切片,kk1为段末坐标
    tpkk = t[kk]
    tpkk1 = t[kk1]

    p = [0 for i in range(2 * m)]
    for ie in range(unit):
        i = lnd[ie][0]  # i.j.k代表节点号
        j = lnd[ie][1]
        k = lnd[ie][2]
        te = tpkk[xy[i][2]][xy[i][3]]
        te += tpkk[xy[j][2]][xy[j][3]]
        te += tpkk[xy[k][2]][xy[k][3]]
        te = te / 3.0
        te1 = tpkk1[xy[i][2]][xy[i][3]]
        te1 += tpkk1[xy[j][2]][xy[j][3]]
        te1 += tpkk1[xy[k][2]][xy[k][3]]
        te1 = te / 3.0
        beta = betaf(ie, te)
        e = Ee(te)
        aa = a(te)
        uu = u(te)
        load = 3 * beta * e * aa * (te - te1) / (2 * (1 - 2 * uu * (1 + 2 * beta)))
        p[2 * i] = p[2 * i] + load * (xy[j][1] - xy[k][1])
        p[2 * i + 1] = p[2 * i + 1] + load * (xy[k][0] - xy[j][0])
        p[2 * j] = p[2 * j] + load * (xy[k][1] - xy[i][1])
        p[2 * j + 1] = p[2 * j + 1] + load * (xy[i][0] - xy[k][0])
        p[2 * k] = p[2 * k] + load * (xy[i][1] - xy[j][1])
        p[2 * k + 1] = p[2 * k + 1] + load * (xy[j][0] - xy[i][0])

    return p


def strain_stress(t, strain):
    if t < 1100:  # 貌似没啥子用
        estrain = 0.000484 - 0.000000368 * t
    else:
        estrain = 0.000147 - 0.00000008 * t
    estrain = 0.00161 - 0.000000756 * t  # 这个有用
    if strain < estrain:
        stress = Ee(t) * strain
    else:
        stress = Ee(t) * estrain + (strain - estrain) * 0.1 * Ee(t)
    return stress


def a_UTS(T):
    # result=-1.8534+45345.6/T*math.exp((-math.log(10,(T/776.5)**2))/0.207368)
    result = -1.8534
    return result


# 求解节点位移
def solver(nx, ny, dx, dy, t, kk, kk1):
    lnd, xy, unit, m = lnd_xy_creack(nx, ny, dx, dy)
    mk1 = mk(unit, m, t, lnd, xy, kk)
    pp = heatload(t, kk, lnd, xy, unit, kk1, m)
    k = np.array(mk1)
    # print(k)
    p = np.array(pp)
    # 求解节点位移
    wx = (np.linalg.pinv(k)).dot(p.reshape(p.shape[0], 1))
    tpkk = t[kk]
    tpkk1 = t[kk1]
    xe = [0 for i in range(6)]  # 创建节点位移
    stress = [0 for i in range(unit)]  # 单元应变
    strain = [0 for i in range(unit)]  # 单元应力
    estress = [[0 for i in range(2)] for j in range(unit)]
    # print(unit)
    for ie in range(unit):
        i = lnd[ie][0]  # i.j.k代表节点号
        j = lnd[ie][1]
        k = lnd[ie][2]
        te = tpkk[xy[i][2]][xy[i][3]]
        te += tpkk[xy[j][2]][xy[j][3]]
        te += tpkk[xy[k][2]][xy[k][3]]
        te = te / 3.0
        te1 = tpkk1[xy[i][2]][xy[i][3]]
        te1 += tpkk1[xy[j][2]][xy[j][3]]
        te1 += tpkk1[xy[k][2]][xy[k][3]]
        te1 = te / 3.0
        b = mb(ie, lnd, xy)  # 3*6的矩阵
        xe[0] = wx[2 * i]
        xe[1] = wx[2 * i + 1]
        xe[2] = wx[2 * j]
        xe[3] = wx[2 * j + 1]
        xe[4] = wx[2 * k]
        xe[5] = wx[2 * k + 1]
        xe = np.array(xe)
        stress[ie] = np.array(b).dot(xe.reshape(xe.shape[0], 1))
        # beta = betaf(ie, te)
        ybx = stress[ie][0]  # x方向的应变
        yby = stress[ie][1]  # y方向的应变
        ybxy = stress[ie][2]  # 切方向的应变
        # ybx=-1*(1-beta)/(1+2*beta)*(ybx+yby)#z方向的应变
        ybz = -u(te) / (1 - u(te)) * (ybx + yby)  # z方向的应变
        # 等效应变
        estress[ie][0] = math.sqrt(
            math.pow(ybx - yby, 2) + math.pow(yby - ybz, 2) + math.pow(ybz - ybx, 2) + 1.5 * math.pow(ybxy, 2)) / (
                                     1 + u(te)) / math.sqrt(2)
        # 等效应力
        estress[ie][1] = strain_stress(te, estress[ie][0])
        d = md(ie, te)
        # strain单元应力
        strain[ie] = np.array(d).dot(stress[ie])
    '''for i in range(m):#'求节点应力应变，相邻单元求平均
        k = 0
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        sum5 = 0
        sum6 = 0
        for ie in range(unit):
            for j in range(2):

                if (i==lnd[ie][j]):
                    k = k+1
                    sum1 = sum1+strain[ie][0]
                    sum2 = sum2+strain[ie][1]
                    sum3 = sum3+strain[ie][2]
                    sum4 = sum4+stress[ie][0]
                    sum5 = sum5+stress[ie][1]
                    sum6 = sum6+stress[ie][2]
        if(k==0):#为了解决一些特殊点
            k=1
        sum1 = sum1 / k / 1000000.0
        sum2 = sum2 / k /1000000.0
        sum3 = sum3 / k / 1000000.0
        sum4 = sum4 / k
        sum5 = sum5 / k
        sum6 = sum6 / k    '''
    max_strain = 0
    x = 0
    y = 0
    for i in range(nx):
        for j in range(ny):
            max_temp = max(strain[i * j][0], strain[i * j][1], strain[i * j][2])
            if (max_temp > max_strain):
                max_strain = max_temp
                x = i
                y = j
    result = max_strain / a_UTS(tpkk[x][y]) * -300000
    # result=pow(10/result,(9-kk1/k))
    # result=10/result
    return stress, estress, strain, result / 20

# stress,estress,strain=solver(20,20,0.1,0.1,t,1805,1900)#计算节点位移
# print(stress[500],estress[500],strain[500])