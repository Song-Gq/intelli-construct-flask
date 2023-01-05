import json
import os
import cv2
import numpy as np
import re
import datetime
import time
from flask import send_file
from alg.wave.wave_svm_detect import wave_start_recog
from excavator import fuzz_index

fnum_dict = {}
fdone_dict = {}
fstatus_dict = {}


def wave_recog(xlsx_path, proc_id, models):
    # path = 'imgs'
    # foldpath = foldread(path)
    # res, mis = imgread(foldpath)
    # recognition not started
    file_num = 1
    fstatus_dict[proc_id] = False
    fnum_dict[proc_id] = file_num
    fdone_dict[proc_id] = 0
    try:
        res = wave_start_recog(xlsx_path, proc_id)
        fnum_dict.pop(proc_id)
        fdone_dict.pop(proc_id)
        fstatus_dict.pop(proc_id)
        with open('statis.json', 'r') as statis_f:
            j = json.load(statis_f)
            j[0]['sum'] = j[0]['sum'] + file_num
            with open('statis.json', 'w') as statis_fw:
                json.dump(j, statis_fw)
        return res
    except Exception as e:
        print("wave_recog(): {}".format(e))
        fnum_dict.pop(proc_id)
        fdone_dict.pop(proc_id)
        fstatus_dict.pop(proc_id)
        return None
