import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from matplotlib.pylab import mpl
from sklearn import model_selection as ms
from sklearn import metrics


def wave_start_recog(xlsx_path, proc_id):
    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

    wavedata = pd.read_excel(xlsx_path)
    WAVE = wavedata['Wavedata']
    WAVE = WAVE.values
    X = WAVE.reshape(1, -1)
    X = X.astype(np.float32)
    svm = joblib.load("alg/wave/wave_svmmodel.m")

    # 输出设备工作状态
    Y_pred = svm.predict(X)
    print(Y_pred)
    if Y_pred == 0:
        print('设备未工作')
    else:
        print('设备正在工作')

    """
    输出波形图和频谱图
    """
    X1 = X
    fft_X1 = fft(X1)
    abs_X1 = np.abs(fft_X1)  # 取复数的绝对值，即复数的模(双边频谱)
    angle_X1 = np.angle(fft_X1)  # 取复数的角度
    N = 24000
    normalization_X1 = abs_X1 / N  # 归一化处理（双边频谱）
    x_fft = normalization_X1[0][0:int(N / 2)]  # 由于对称性，只取一半区间（单边频谱）
    # x = x.reshape(1, -1)
    xlabel = np.arange(N)
    half_xlabel = xlabel[range(int(N / 2))]

    # 输出原始波形
    plt.figure(1)
    plt.plot(xlabel, WAVE)
    plt.ylim(-0.01, 0.01)
    plt.title('Original Wave')

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 14)
    plt.savefig('alg/wave/output/original' + str(proc_id) + '.jpg')
    # plt.show()
    # plt.close()

    # 输出单边振幅谱(归一化)
    plt.figure(2)
    plt.plot(half_xlabel, x_fft, 'blue')
    plt.ylim(0,5e-5)
    plt.title('Processed', fontsize=9, color='blue')

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 14)
    plt.savefig('alg/wave/output/processed' + str(proc_id) + '.jpg')
    # plt.show()
    # plt.close()

    return [int(Y_pred[0]),
            [xlabel.tolist(), WAVE.tolist()],
            [half_xlabel.tolist(), x_fft.tolist()]]
