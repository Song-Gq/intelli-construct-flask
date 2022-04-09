import easyocr
import os
import cv2
import numpy as np
import re
import xlwt
import datetime


def foldread(prepath: str):
    print('开始寻找图片目录')
    filetype = ['jpg', 'png']
    foldname = []

    def bfs(prepath, foldname):
        if filetype[0] in os.listdir(prepath)[0] or filetype[1] in os.listdir(prepath)[0]:
            foldname.append(prepath[:])
            return
        for k, filename in enumerate(os.listdir(prepath)):
            curpath = prepath + './' + filename
            bfs(curpath, foldname)

    bfs(prepath, foldname)
    return foldname


# 从文件列表中读取图片,识别图片，[姓名，采样和检测，采样、检测结果，日期，来源？]
def f_imgread(flist):
    print('开始读取图片并识别')
    info = ['类型', '姓名', '采样时间', '检测结果']
    res = []
    mis = []
    reader = easyocr.Reader(['ch_sim', 'en'])  # this needs to run only once to load the model into memory
    resi = []
    for f in flist:
        resj = [f]
        img = cv2.imdecode(np.asarray(bytearray(flist[f].read()), dtype="uint8"), cv2.IMREAD_COLOR)
        img_array = np.array(img)
        result = reader.readtext(img_array, detail=0)
        iname = result.index('姓名')
        if '\u4e00' <= result[iname + 1][0] <= '\u9fff':
            iname += 1
        else:
            iname += 2
        name = result[iname]
        itime = result.index('采样时间')
        time = result[itime + 1]
        ire = result.index('检测结果')
        re = result[ire + 1]
        if '上传' in re:
            f = '已采样'
            c = '采样'
        if '阴' in re:
            f = '已检测' + re
            c = '检测'
        elif '阳' in re:
            f = '已检测' + re
            c = '检测'
            mis.append([c, name, time, f])
        resj += [c, name, time, f]
        if len(resj) == 5:
            resi.append(resj)
        else:
            mis.append(resj)
    res.append(resi)
    return res, mis


# 从文件夹列表中读取图片,识别图片，[姓名，采样和检测，采样、检测结果，日期，来源？]
def imgread(foldpath: str):
    print('开始读取图片并识别')
    info = ['图片路径', '类型', '姓名', '采样时间', '检测结果']
    m = len(foldpath)
    res = []
    mis = []
    reader = easyocr.Reader(['ch_sim', 'en'])  # this needs to run only once to load the model into memory
    for i in range(m):
        fold = foldpath[i]
        filelist = os.listdir(fold)
        resi = []
        for filename in filelist:
            file = fold + './' + filename
            resj = [file]
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
            img_array = np.array(img)
            result = reader.readtext(img_array, detail=0)
            iname = result.index('姓名')
            if '\u4e00' <= result[iname + 1][0] <= '\u9fff':
                iname += 1
            else:
                iname += 2
            name = result[iname]
            itime = result.index('采样时间')
            time = result[itime + 1]
            ire = result.index('检测结果')
            re = result[ire + 1]
            if '上传' in re:
                f = '已采样'
                c = '采样'
            if '阴' in re:
                f = '已检测' + re
                c = '检测'
            elif '阳' in re:
                f = '已检测' + re
                c = '检测'
                mis.append([fold, c, name, time, f])
            resj += [c, name, time, f]
            if len(resj) == 5:
                resi.append(resj)
            else:
                mis.append(resj)
        res.append(resi)
    return res, mis


# #输出为EXCEL
def Toxls(res, mis, info):
    print('开始生成检测统计报告')
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Sheet Name1")
    for i in range(len(info)):
        sheet.write(0, i, info[i])
    c = 1
    for fold in res:
        for per in fold:
            for j in range(len(per)):
                sheet.write(c, j, per[j])
            c += 1

    curr_time = datetime.date.today()
    excel_filename = str(curr_time) + '核酸采样结果' + '.xls'
    workbook.save(excel_filename)
    print('生成检测报告为' + excel_filename)


def start_recognition(files):
    # path = 'imgs'
    # foldpath = foldread(path)
    # res, mis = imgread(foldpath)
    try:
        res, mis = f_imgread(files)
        if mis:
            print('存在异常检测结果', mis)
        info = ['图片路径', '类型', '姓名', '采样时间', '检测结果']
        Toxls(res, mis, info)
    except Exception as e:
        print(e)


# if __name__ == '__main__':
#     path = 'imgs'
#     foldpath = foldread(path)
#     res, mis = imgread(foldpath)
#     if mis:
#         print('存在异常检测结果', mis)
#     info = ['图片路径', '类型', '姓名', '采样时间', '检测结果']
#     Toxls(res, mis, info)
