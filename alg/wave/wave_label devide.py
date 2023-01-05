"""
对文件进行重命名
"""
# remane
import os
def openreadtxt(file_name):
    data = []
    file = open(file_name,'r', encoding='utf-8')  #打开文件
    file_data = file.readlines() #读取所有行
    for row in file_data:
        tmp_list = row.split(' ') #按‘，’切分每行的数据
        tmp_list[-1] = tmp_list[-1].replace('\n', ',') #去掉换行符
        tmp_list[0] = tmp_list[0].replace('\t', ',')  # 去掉换行符
        data.append(tmp_list) #将每行数据插入data中
    return data
data = openreadtxt('1.txt')
print(data)
for i in range(len(data)):
    C = str(i+8) + '-' +data[i][0].strip(',') + '/'
    C1 = data[i][0].strip(',')
    image_dir = 'F:/智慧工地视频数据集/1-智慧工地视频/工地视频素材（已整理）/' + C  # 源图片路径
    images_list = os.listdir(image_dir)
    nums = len(os.listdir(image_dir))
    # output_dir = "F:\\1-Smart Site dataset\\Dangerous behaviors dataset\\" + C # 图像重命名后的保存路径
    output_dir = image_dir
    start = 1

    for files in images_list:
        if files[-3:-1] =='mp' or files[-3:-1] =='MP':
            os.rename(image_dir + files, output_dir + C1 + '_' + str(start) + '.mp4')  # 前面是旧的路径,后面是新路径
            start = start + 1

    print(files)
    print('found %d pictures' % (int(start)-1))

print('finished!')


"""
根据文件名及标签对文件进行分类
"""
##  label devide
import shutil
import os
import pandas as pd

#读取工作簿和工作簿中的工-作表
wave_data = pd.read_excel('E:\\2-Algorithm\\facepose\\FACEpose\\results\\wave_dataset\\label.xlsx')
name = wave_data['name']
work_state = wave_data['work_state']

target_path0 = '/results/WAV/static/'     #保存文件夹地址
target_path1 = '/results/WAV/work/'     #保存文件夹地址
original_path = 'results/WAV/image/'       #需查找图片的原始地址

for i in range((len(wave_data))):
    row = name[i] + '.jpg'
    if os.path.exists(target_path0 + row) or os.path.exists(target_path1 + row) :
        print("已存在文件")
    else:
        full_path = original_path + row   #还没有
        if work_state[i] == 0:
            shutil.move(full_path, target_path0 +row)
        if work_state[i] == 1:
            shutil.move(full_path, target_path1 +row)