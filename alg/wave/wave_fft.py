import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

path = "C:/Users/17389/Desktop/wave_2022_070000_075959.txt"
data = np.genfromtxt(path, dtype=[object, object, float])  # 将文件中数据加载到data数组里

#
# print(data[0])
# print(str(data[0][0]))
# data[0][0] = str(data[0][0], encoding='utf-8')
# data[0][1] = str(data[0][1], encoding='utf-8')
#
# data = pd.DataFrame(data[0:30])
#
# writer = pd.ExcelWriter('A00.xlsx')		# 写入Excel文件
# data.to_excel(writer, 'page_1', float_format='%.6f')		# ‘page_1’是写入excel的sheet名
# writer.save()
#
# writer.close()

# print(data[0][0][1:10])
# print(type(data))
# print(type(data[0]))
# print(type(data[0][1]))

# data = pd.DataFrame(data)
#
# writer = pd.ExcelWriter('A.xlsx')		# 写入Excel文件
# data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
# writer.save()
#
# writer.close()


X = []
for i in range(len(data)):
    X.append(data[i][2])
print(len(X))
Y = [[0 for k1 in range(24000)] for k2 in range(30)]

for j1 in range(30):
    for j2 in range(24000):
        Y[j1][j2] = X[j1 * 24000 + j2]

y0 = Y[0]
fft_y0 = fft(y0)  # 快速傅里叶变换
N = 24000
x0 = np.arange(N)  # 频率个数
half_x0 = x0[range(int(N / 2))]  # 取一半区间

abs_y0 = np.abs(fft_y0)  # 取复数的绝对值，即复数的模(双边频谱)
angle_y0 = np.angle(fft_y0)  # 取复数的角度
normalization_y0 = abs_y0 / N  # 归一化处理（双边频谱）
normalization_half_y0 = normalization_y0[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）


# plt.subplot(211)
for j in range(1):
    n = 2*j
    # Y[j] = X[n * 60 * 200:(n + 2) * 60 * 200]
    y = Y[j]
    fft_y = fft(y)  # 快速傅里叶变换
    N = 24000
    x = np.arange(N)  # 频率个数
    half_x = x[range(int(N / 2))]  # 取一半区间

    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    angle_y = np.angle(fft_y)  # 取复数的角度
    # normalization_y = abs_y / N  # 归一化处理（双边频谱）
    normalization_y = abs_y / N - normalization_y0  # 归一化处理（双边频谱）
    normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

    plt.figure(1)
    plt.subplot(211)
    plt.plot(x, y)
    plt.ylim(-0.01, 0.01)
    plt.title('原始波形')

    plt.subplot(212)
    plt.plot(half_x, normalization_half_y, 'blue')
    plt.ylim(0,5e-5)
    plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 14)
    # plt.savefig('results/fft_wave/chushi/第{}到{}分钟波形000.jpg'.format(n, n+2), dpi=100)
    # plt.savefig('results/fft_wave/第{}到{}分钟波形1.jpg'.format(n, n+2))
    # plt.show()
    # plt.close()

    plt.figure(2)
    plt.plot(half_x, normalization_half_y, 'blue')
    plt.ylim(0,5e-5)
    plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
    plt.show()

# # plt.subplot(231)
# for j in range(30):
#     n = 2*j
#     # Y[j] = X[n * 60 * 200:(n + 2) * 60 * 200]
#     y = Y[j]
#     fft_y = fft(y)  # 快速傅里叶变换
#     N = 24000
#     x = np.arange(N)  # 频率个数
#     half_x = x[range(int(N / 2))]  # 取一半区间
#
#     abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
#     angle_y = np.angle(fft_y)  # 取复数的角度
#     # normalization_y = abs_y / N  # 归一化处理（双边频谱）
#     normalization_y = abs_y / N - normalization_y0  # 归一化处理（双边频谱）
#     normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
#
#     plt.subplot(231)
#     plt.plot(x, y)
#     # plt.ylim(-4e-3, 4e-3)
#     plt.ylim(-0.01, 0.01)
#     plt.title('原始波形')
#
#     plt.subplot(232)
#     plt.plot(x, fft_y, 'black')
#     plt.ylim(-0.6, 0.6)
#     plt.title('双边振幅谱(未求振幅绝对值)', fontsize=9, color='black')
#
#     plt.subplot(233)
#     plt.plot(x, abs_y, 'r')
#     plt.ylim(0, 0.7)
#     plt.title('双边振幅谱(未归一化)', fontsize=9, color='red')
#
#     plt.subplot(234)
#     plt.plot(x, angle_y, 'violet')
#     plt.title('双边相位谱(未归一化)', fontsize=9, color='violet')
#
#     plt.subplot(235)
#     plt.plot(x, normalization_y, 'g')
#     plt.ylim(0, 3e-5)
#     plt.title('双边振幅谱(归一化)', fontsize=9, color='green')
#
#     plt.subplot(236)
#     plt.plot(half_x, normalization_half_y, 'blue')
#     plt.ylim(0,5e-5)
#     plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
#
#     figure = plt.gcf()  # get current figure
#     figure.set_size_inches(18, 12)
#     plt.savefig('results/fft_wave/chushi/第{}到{}分钟波形000.jpg'.format(n, n+2), dpi=100)
#     # plt.savefig('results/fft_wave/第{}到{}分钟波形1.jpg'.format(n, n+2))
#     plt.show()
#     # plt.close()
