import json
from alg.equip.yolo_openpose import equip_start_recog

fnum_dict = {}
fdone_dict = {}
fstatus_dict = {}


def equip_recog(img_path, proc_id):
    file_num = 1
    fstatus_dict[proc_id] = False
    fnum_dict[proc_id] = file_num
    fdone_dict[proc_id] = 0
    try:
        res = equip_start_recog(img_path, proc_id)
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
        print("equip_recog(): {}".format(e))
        fnum_dict.pop(proc_id)
        fdone_dict.pop(proc_id)
        fstatus_dict.pop(proc_id)
        return None
