import pandas as pd
import os
import math
import numpy as np
import itertools


def Distance(Ax,Ay,Bx,By):
    distance=[]
    for i in range(len(Ax)):
        d=math.sqrt((Ax[i]-Bx[i])**2+(Ay[i]-By[i])**2)
        distance.append(d)
    return distance


def Angle(Ax,Ay,Bx,By,Cx,Cy):
    angle=[]
    for i in range(len(Ax)):
        a=math.sqrt((Bx[i]-Cx[i])**2+(By[i]-Cy[i])**2)
        b = math.sqrt((Ax[i] - Cx[i]) ** 2 + (Ay[i] - Cy[i]) ** 2)
        c = math.sqrt((Bx[i] - Ax[i]) ** 2 + (By[i] - Ay[i]) ** 2)
        B=math.degrees(math.acos((a**2+c**2-b**2)/(2*a*c)))
        angle.append(B)
    return angle


def findMaxConsecutiveOnes(nums):
    result_list, max_nums = [], 0
    new_nums = [0] + nums + [0]
    max_num=0
    for i in range(len(new_nums)):
        if new_nums[i] == 0:
            result_list.append(i)
    for j in range(1, len(result_list)):
        max_num = max(max_num, result_list[j] - result_list[j - 1] - 1)
    return max_num


def driver_action_start_recog(json_path, proc_id):
    LEarX=[]
    LEarY=[]

    REarX=[]
    REarY=[]

    LWristX=[]
    LWristY=[]

    LElbowX=[]
    LElbowY=[]

    LShoulderX=[]
    LShoulderY=[]

    RWristX=[]
    RWristY=[]

    RElbowX=[]
    RElbowY=[]

    RShoulderX=[]
    RShoulderY=[]

    #读取骨架信息json文件
    data = pd.read_json(json_path)

    for i in range(len(data['keypoints'])):
        LEarX.append(data['keypoints'][i][9])
        LEarY.append(data['keypoints'][i][10])

        REarX.append(data['keypoints'][i][12])
        REarY.append(data['keypoints'][i][13])

        LWristX.append(data['keypoints'][i][27])
        LWristY.append(data['keypoints'][i][28])

        LElbowX.append(data['keypoints'][i][21])
        LElbowY.append(data['keypoints'][i][22])

        LShoulderX.append(data['keypoints'][i][15])
        LShoulderY.append(data['keypoints'][i][16])

        RWristX.append(data['keypoints'][i][30])
        RWristY.append(data['keypoints'][i][31])

        RElbowX.append(data['keypoints'][i][24])
        RElbowY.append(data['keypoints'][i][25])

        RShoulderX.append(data['keypoints'][i][18])
        RShoulderY.append(data['keypoints'][i][19])

    #下面四个值可以画四张图分别展示
    D_RWri_REar=Distance(RWristX,RWristY,REarX,REarY)#手腕到耳朵的距离
    D_LWri_LEar=Distance(LWristX,LWristY,LEarX,LEarY)

    A_RWri_RElb_RSho=Angle(RWristX,RWristY,RElbowX,RElbowY,RShoulderX,RShoulderY)#手腕-手肘-肩形成的夹角
    A_LWri_LElb_LSho=Angle(LWristX,LWristY,LElbowX,LElbowY,LShoulderX,LShoulderY)

    res = []
    for i in range(len(data['keypoints'])):
        res.append([i, D_RWri_REar[i], D_LWri_LEar[i], A_RWri_RElb_RSho[i], A_LWri_LElb_LSho[i]])

    times=[]
    for i in range(len(D_RWri_REar)):
        if i%5==0:#每5帧判断一次半秒
            if D_RWri_REar[i]>=57 or D_RWri_REar[i]<=77 or D_LWri_LEar[i]>=57 or D_LWri_LEar[i]<=77:
                if A_RWri_RElb_RSho[i]<=30 or A_LWri_LElb_LSho[i]<=30:
                    times.append(1)
                else:
                    times.append(0)
            else:
                times.append(0)
    print(times)
    A=findMaxConsecutiveOnes(times)#连续1的最大个数

    print(A)
    #下面这个需要展示的结果
    if A>=11:
        print('Phoning')
        phoning = 1
    else:
        print('Not Phoning')
        phoning = 0
    # if times>=10:
    #     print('Phoning')

    return [phoning, res]

    # print('a')
    # print(D_RWri_REar)
    # print('b')
    # print(D_LWri_LEar)
    # print('c')
    # print(A_RWri_RElb_RSho)
    # print('d')
    # print(A_LWri_LElb_LSho)
