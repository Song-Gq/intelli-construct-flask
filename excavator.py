import json
import os
import cv2
import numpy as np
import re
import datetime
import time
from flask import send_file
from alg.excavator.yolov3_deepsort import excavator_start_recog

fnum_dict = {}
fdone_dict = {}
fstatus_dict = {}


def fuzz_index(in_list, target_str):
    target_list = [i for i, x in enumerate(in_list) if x.find(target_str) != -1]
    return target_list[0]


# 从文件列表中读取图片,识别图片，[姓名，采样和检测，采样、检测结果，日期，来源？]
# def f_imgread(flist, proc_id, reader):
#     fstatus_dict[proc_id] = True
#     print('开始读取图片并识别')
#     info = ['类型', '姓名', '采样时间', '检测结果']
#     res = []
#     mis = []
#     resi = []
#     for f in flist:
#         try:
#             resj = []
#             img = cv2.imdecode(np.asarray(bytearray(flist[f].read()), dtype="uint8"), cv2.IMREAD_COLOR)
#             img_array = np.array(img)
#             result = reader.readtext(img_array, detail=0)
#             iname = fuzz_index(result, '姓名')
#             # iname = result.index('姓名')
#             if '\u4e00' <= result[iname + 1][0] <= '\u9fff':
#                 iname += 1
#             else:
#                 iname += 2
#             name = result[iname]
#             itime = fuzz_index(result, '采样时间')
#             # itime = result.index('采样时间')
#             t = format_time(result[itime + 1])
#             ire = fuzz_index(result, '检测结果')
#             # ire = result.index('检测结果')
#             r = result[ire + 1]
#             c = ""
#             if '上传' in r:
#                 f = '已采样'
#                 c = '采样'
#             if '阴' in r:
#                 f = '已检测' + r
#                 c = '检测'
#             elif '阳' in r:
#                 f = '已检测' + r
#                 c = '检测'
#                 mis.append([c, name, t, f])
#             resj = [c, name, t, f]
#             if len(resj) == 4:
#                 resi.append(resj)
#             else:
#                 mis.append(resj)
#             fdone_dict[proc_id] = fdone_dict[proc_id] + 1
#         except Exception as e:
#             print("f_imgread(): {}\nfile: {}".format(e, f))
#             fname = f.split('=')[2]
#             misf = ["未知图片", fname, " ", "检测失败"]
#             mis.append(misf)
#             resi.append(misf)
#     res.append(resi)
#     return res, mis


def excavator_recog(file_path, proc_id):
    # path = 'imgs'
    # foldpath = foldread(path)
    # res, mis = imgread(foldpath)
    # recognition not started
    file_num = 1
    fstatus_dict[proc_id] = False
    fnum_dict[proc_id] = file_num
    fdone_dict[proc_id] = 0
    try:
        res, state = excavator_start_recog(file_path, proc_id)
        # res, mis = f_imgread(files, proc_id, reader)
        # if mis:
        #     print('存在异常检测结果', mis)
        # info = ['类型', '姓名', '采样时间', '检测结果']
        # Toxls(res, mis, info, proc_id)
        fnum_dict.pop(proc_id)
        fdone_dict.pop(proc_id)
        fstatus_dict.pop(proc_id)
        with open('statis.json', 'r') as statis_f:
            j = json.load(statis_f)
            j[0]['sum'] = j[0]['sum'] + file_num
            with open('statis.json', 'w') as statis_fw:
                json.dump(j, statis_fw)
        return res, state
    except Exception as e:
        print("excavator_recog(): {}".format(e))
        fnum_dict.pop(proc_id)
        fdone_dict.pop(proc_id)
        fstatus_dict.pop(proc_id)
        return None
