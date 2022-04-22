import easyocr
import os
import cv2
import numpy as np
import re
import xlwt
import datetime
import time

from flask import send_file

fnum_dict = {}
fdone_dict = {}
fstatus_dict = {}


def fuzz_index(in_list, target_str):
    target_list = [i for i, x in enumerate(in_list) if x.find(target_str) != -1]
    return target_list[0]


# def foldread(prepath: str):
#     print('开始寻找图片目录')
#     filetype = ['jpg', 'png']
#     foldname = []
#
#     def bfs(prepath, foldname):
#         if filetype[0] in os.listdir(prepath)[0] or filetype[1] in os.listdir(prepath)[0]:
#             foldname.append(prepath[:])
#             return
#         for k, filename in enumerate(os.listdir(prepath)):
#             curpath = prepath + './' + filename
#             bfs(curpath, foldname)
#
#     bfs(prepath, foldname)
#     return foldname


def format_time(t_str):
    try:
        t_str = t_str.replace('l', '1')
        t_str = t_str.replace('/', '1')
        t_str = t_str.replace('\\', '1')
        t_str = t_str.replace('|', '1')
        t_str = t_str.replace(" ", "")
        t_str = t_str.replace(".", ":")
        t_str = datetime.datetime.strptime(t_str, "%Y-%m-%d%X")
        return t_str.strftime("%Y-%m-%d %X")
    except Exception as e:
        print(e)
        return t_str


# 从文件列表中读取图片,识别图片，[姓名，采样和检测，采样、检测结果，日期，来源？]
def f_imgread(flist, proc_id):
    fstatus_dict[proc_id] = True
    print('开始读取图片并识别')
    info = ['类型', '姓名', '采样时间', '检测结果']
    res = []
    mis = []
    reader = easyocr.Reader(['ch_sim', 'en'])  # this needs to run only once to load the model into memory
    resi = []
    for f in flist:
        try:
            resj = []
            img = cv2.imdecode(np.asarray(bytearray(flist[f].read()), dtype="uint8"), cv2.IMREAD_COLOR)
            img_array = np.array(img)
            result = reader.readtext(img_array, detail=0)
            iname = fuzz_index(result, '姓名')
            # iname = result.index('姓名')
            if '\u4e00' <= result[iname + 1][0] <= '\u9fff':
                iname += 1
            else:
                iname += 2
            name = result[iname]
            itime = fuzz_index(result, '采样时间')
            # itime = result.index('采样时间')
            t = format_time(result[itime + 1])
            ire = fuzz_index(result, '检测结果')
            # ire = result.index('检测结果')
            r = result[ire + 1]
            c = ""
            if '上传' in r:
                f = '已采样'
                c = '采样'
            if '阴' in r:
                f = '已检测' + r
                c = '检测'
            elif '阳' in r:
                f = '已检测' + r
                c = '检测'
                mis.append([c, name, t, f])
            resj = [c, name, t, f]
            if len(resj) == 4:
                resi.append(resj)
            else:
                mis.append(resj)
            fdone_dict[proc_id] = fdone_dict[proc_id] + 1
        except Exception as e:
            print("f_imgread(): {}\nfile: {}".format(e, f))
            fname = f.split('=')[2]
            misf = ["未知图片", fname, " ", "检测失败"]
            mis.append(misf)
            resi.append(misf)
    res.append(resi)
    return res, mis


# 从文件夹列表中读取图片,识别图片，[姓名，采样和检测，采样、检测结果，日期，来源？]
# def imgread(foldpath: str):
#     print('开始读取图片并识别')
#     info = ['图片路径', '类型', '姓名', '采样时间', '检测结果']
#     m = len(foldpath)
#     res = []
#     mis = []
#     reader = easyocr.Reader(['ch_sim', 'en'])  # this needs to run only once to load the model into memory
#     for i in range(m):
#         fold = foldpath[i]
#         filelist = os.listdir(fold)
#         resi = []
#         for filename in filelist:
#             file = fold + './' + filename
#             resj = [file]
#             img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
#             img_array = np.array(img)
#             result = reader.readtext(img_array, detail=0)
#             iname = result.index('姓名')
#             if '\u4e00' <= result[iname + 1][0] <= '\u9fff':
#                 iname += 1
#             else:
#                 iname += 2
#             name = result[iname]
#             itime = result.index('采样时间')
#             time = result[itime + 1]
#             ire = result.index('检测结果')
#             re = result[ire + 1]
#             if '上传' in re:
#                 f = '已采样'
#                 c = '采样'
#             if '阴' in re:
#                 f = '已检测' + re
#                 c = '检测'
#             elif '阳' in re:
#                 f = '已检测' + re
#                 c = '检测'
#                 mis.append([fold, c, name, time, f])
#             resj += [c, name, time, f]
#             if len(resj) == 5:
#                 resi.append(resj)
#             else:
#                 mis.append(resj)
#         res.append(resi)
#     return res, mis


# #输出为EXCEL
def Toxls(res, mis, info, proc_id):
    print('开始生成检测统计报告')
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Sheet1")
    for i in range(len(info)):
        sheet.write(0, i, info[i])
    c = 1
    for fold in res:
        for per in fold:
            for j in range(len(per)):
                sheet.write(c, j, per[j])
            c += 1

    # curr_time = datetime.datetime.now().strftime("%Y-%m-%d %X")
    excel_filename = str(proc_id)
    workbook.save("out_excel/{}.xls".format(excel_filename))
    print("生成检测报告: out_excel/{}.xls".format(excel_filename))


def start_recognition(files, proc_id):
    # path = 'imgs'
    # foldpath = foldread(path)
    # res, mis = imgread(foldpath)
    # recognition not started
    fstatus_dict[proc_id] = False
    fnum_dict[proc_id] = len(files)
    fdone_dict[proc_id] = 0
    try:
        res, mis = f_imgread(files, proc_id)
        if mis:
            print('存在异常检测结果', mis)
        info = ['类型', '姓名', '采样时间', '检测结果']
        Toxls(res, mis, info, proc_id)
        fnum_dict.pop(proc_id)
        fdone_dict.pop(proc_id)
        fstatus_dict.pop(proc_id)
        return res[0], mis
    except Exception as e:
        print("start_recognition(): {}".format(e))
        fnum_dict.pop(proc_id)
        fdone_dict.pop(proc_id)
        fstatus_dict.pop(proc_id)
        return None, None


def get_prog(proc_id):
    try:
        if proc_id not in fnum_dict.keys():
            # token valid but upload not done
            return -2
        if not fstatus_dict[proc_id]:
            # recognition not started
            return -2
        if fnum_dict[proc_id] == 0:
            return -1
        return fdone_dict[proc_id] / fnum_dict[proc_id]
    except Exception as e:
        print("get_prog() in start_recognition.py: {}".format(e))
        return -1


def get_excel(proc_id):
    file_path = "out_excel/{}.xls".format(proc_id)
    if os.path.exists(file_path):
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d %X")
        return send_file(file_path, as_attachment=True,
                         attachment_filename="核酸检测识别报告-{}.xls".format(curr_time))
    return None


# if __name__ == '__main__':
#     path = 'imgs'
#     foldpath = foldread(path)
#     res, mis = imgread(foldpath)
#     if mis:
#         print('存在异常检测结果', mis)
#     info = ['图片路径', '类型', '姓名', '采样时间', '检测结果']
#     Toxls(res, mis, info)
