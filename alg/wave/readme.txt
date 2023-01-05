数据处理文件：
【wave_dataset.py】生成wave数据集、图片集
【wave_fft.py】对数据做fft变换，将时域上的数据转换到频域上
【wave_hotpotints.py】将一维数据变成二维的热力图
【wave_label devide.py】对文件进行重命名、根据文件名及标签对文件进行分类

SVM模型
【wave_svm_model.py】时域频域精度对比、在声纹的时域数据集上训练模型保存为【wave_svmmodel.m】

输出结果
【wave_svm_detect.py】
① 输出设备工作状态（设备未工作、设备正在工作）
② 输出波形图和频谱图，保存到outcomes文件夹
测试数据集在results/WAV/data下
