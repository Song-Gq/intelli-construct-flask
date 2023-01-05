import numpy as np
import cv2
from sklearn import model_selection as ms
from sklearn import metrics
import pandas as pd
from matplotlib.pylab import mpl
import joblib
from scipy.fftpack import fft
from sklearn import svm

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

# 加载声纹数据
path = "dataset/waves/CH.txt"
wavedata = np.genfromtxt(path, dtype=[object, object, float])
# 加载label
wavedata_label = pd.read_excel("dataset/waves/label.xlsx")

X = np.zeros([int(len(wavedata)/24000), 24000])
X = X.astype(np.float32)
label = wavedata_label['work_state']
Y = label.values

for i in range(int(len(wavedata)/24000)):
    for j in range(24000):
        X[i][j] = wavedata[i*24000+j][2]


"""
时域频域精度对比
"""
kernels = [cv2.ml.SVM_LINEAR,cv2.ml.SVM_INTER,
           cv2.ml.SVM_SIGMOID,cv2.ml.SVM_RBF]

# fft变换
fft_X = fft(X)
abs_X = np.abs(fft_X)  # 取复数的绝对值，即复数的模(双边频谱)
angle_X = np.angle(fft_X)  # 取复数的角度
N = 24000
normalization_X = abs_X / N  # 归一化处理（双边频谱）
x = normalization_X[0:int(N / 2)]  # 由于对称性，只取一半区间（单边频谱）

print('频域')
#  分割数据集，20%为测试集
x_train,x_text,y_train,y_text = ms.train_test_split(x, Y,test_size=0.2,random_state=42)
#  训练含有不同核的svm分类器
for i,kernel in enumerate(kernels):
    svm = cv2.ml.SVM_create()
    #  设置svm核
    svm.setKernel(kernel)
    svm.train(x_train,cv2.ml.ROW_SAMPLE,y_train)
    b,y_pred = svm.predict(x_text)
    #  用测试集计算准确率
    a=metrics.accuracy_score(y_text,y_pred)
    print('y_text：{},准确率：{}'.format(y_text,a))

print('时域')
X_train,X_text,Y_train,Y_text = ms.train_test_split(X, Y,test_size=0.2,random_state=42)
#  训练含有不同核的svm分类器
for i,kernel in enumerate(kernels):
    svm = cv2.ml.SVM_create()
    #  设置svm核
    svm.setKernel(kernel)
    svm.train(X_train,cv2.ml.ROW_SAMPLE,Y_train)
    b,Y_pred = svm.predict(X_text)
    #  用测试集计算准确率
    a=metrics.accuracy_score(Y_text,Y_pred)
    print('Y_text：{}，准确率：{}'.format(Y_text,a))


# # 在声纹的时域数据集上训练模型保存并保存
clf = svm.SVC()
clf.fit(X_train, Y_train)
joblib.dump(clf, "wave_svmmodel.m")