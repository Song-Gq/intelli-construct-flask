"""
生成wave数据集、图片集
"""
import os
import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import math

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

path = "dataset/waves/CH3.txt"
wavedata = np.genfromtxt(path, dtype=[object, object, float])  # 将文件中数据加载到data数组里
A = [0 for k1 in range(int(len(wavedata)/24000))]
CH = np.array(['CH3' for _ in range(1) for _ in range(24000)])
CH = pd.DataFrame(CH)

# path = E:/2-Algorithm/facepose/FACEpose/results/WAV
for i in range(len(wavedata)):
    wavedata[i][0] = str(wavedata[i][0], encoding='utf-8')
    wavedata[i][1] = str(wavedata[i][1], encoding='utf-8')
    i1 = 0

    if i % 24000 == 0 and i% 720000 !=0 or (i+1) % 720000 == 0:
        if (i+1) % 720000 == 0:
            i1 = 1
        A[int(i/24000)-1+i1] = ('CH3_{}点_第{}到{}分钟'.format(wavedata[i][1][0:2], int(wavedata[i][1][3:5])-2+i1, int(wavedata[i][1][3:5])+i1))		# 写入Excel文件

        # 生成数据集
        Data_2min = pd.DataFrame(wavedata[i-24000:i])
        Data_2min_CH = pd.concat([CH, Data_2min], axis=1)
        writer = pd.ExcelWriter('results/WAV/data/CH3_{}点_第{}到{}分钟.xlsx'.format(wavedata[i][1][0:2], int(wavedata[i][1][3:5])-2+i1, int(wavedata[i][1][3:5])+i1))		# 写入Excel文件
        Data_2min_CH.to_excel(writer, 'wave_dataset',header=['Channel', 'Day','Time','Wavedata'] , float_format='%.6f')		# ‘page_1’是写入excel的sheet名
        writer.save()
        writer.close()

        # 生成波形图和频谱图
        y = [0 for k1 in range(24000)]
        for j2 in range(24000):
            y[j2] = (float(wavedata[i-24000 + j2][2]))

        fft_y = fft(y)  # 快速傅里叶变换
        N = 24000
        x = np.arange(N)  # 频率个数
        half_x = x[range(int(N / 2))]  # 取一半区间

        abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
        angle_y = np.angle(fft_y)  # 取复数的角度
        normalization_y = abs_y / N  # 归一化处理（双边频谱）
        normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

        # 双图
        # plt.figure(1)
        # plt.subplot(211)
        # plt.plot(x, y)
        # plt.ylim(-0.01, 0.01)
        # plt.title('原始波形')
        #
        # plt.subplot(212)
        # plt.plot(half_x, normalization_half_y, 'blue')
        # plt.ylim(0,5e-5)
        # plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
        #
        # figure = plt.gcf()  # get current figure
        # figure.set_size_inches(18, 14)
        # plt.savefig('results/wave_dataset/image2/CH3_{}点_第{}到{}分钟_2.jpg'.format(wavedata[i][1][0:2], int(wavedata[i][1][3:5])-2+i1, int(wavedata[i][1][3:5])+i1))
        # plt.close()

        plt.figure(2)
        plt.plot(half_x, normalization_half_y, 'blue')
        plt.ylim(0,5e-5)
        plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(18, 14)
        plt.savefig('results/WAV/data/CH3_{}点_第{}到{}分钟.jpg'.format(wavedata[i][1][0:2], int(wavedata[i][1][3:5])-2+i1, int(wavedata[i][1][3:5])+i1))
        plt.close()
        print('第{}次迭代'.format(int(i/24000)))
        print(A[int(i/24000)-1+i1])
print(i)
List = pd.DataFrame(A)
writer = pd.ExcelWriter('results/wave_dataset/label2.xlsx')		# 写入Excel文件
List.to_excel(writer, 'wave_dataset',header=['name'], float_format='%.6f')		# ‘page_1’是写入excel的sheet名
writer.save()
writer.close()