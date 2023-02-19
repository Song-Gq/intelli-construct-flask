# from IOU_3_tree import YOLO
from yolo import YOLO
from PIL import Image
import tensorflow as tf
import os
import cv2
from timeit import default_timer as timer
import numpy as np
import glob

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
yolo = YOLO()

# # 读取文件夹
# path = "C:\\Users\\HP\\Desktop\\测试集的输出结果\\000655.jpg"
# outdir = "C:\\Users\\HP\\Desktop\\测试集的输出结果\\result_2022"
#
# for jpgfile in glob.glob(path):
#     image = Image.open(jpgfile)
#     image = yolo.detect_image(image)
#     # img.show()
#     # image.save(os.path.join(outdir, os.path.basename(jpgfile)))  # 保存识别之后的图片

# path = ".\\000312.jpg"
# outdir = ".\\trb_out"
#
# for jpgfile in glob.glob(path):
#     image = Image.open(jpgfile)
#     image = yolo.detect_image(image)
#     image.show()
#     image.save(os.path.join(outdir, os.path.basename(jpgfile)))  # 保存识别之后的图片

# 输入图片
while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.show()


# 检测视频
# import glob # 视频检测代码
# # cap = cv2.VideoCapture(0) # 如果要调用笔记本摄像头进行目标检测的话，执行这行代码
# cap = cv2.VideoCapture('D:\\大四下资料\\1毕业设计\\深度学习资料\\工地监控视频\\1.mp4')  # 如果使用的电脑没有摄像头的话，就执行这行代码，该代码的功能是识别保存好的视频
# ret, frame = cap.read() # 读取视频流
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
# save_video = '300' + '.mp4'  # 视频保存名称
# out = cv2.VideoWriter(save_video, fourcc, 20.0, (1000, 550))  # 保存识别之后的视频
# video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))  # 视频参数
# video_fps = cap.get(cv2.CAP_PROP_FPS)  # 视频参数
# video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 视频参数
# # isOutput = True if output_path != "" else False
# accum_time = 0
# curr_fps = 0
# fps = "FPS: ??"
# prev_time = timer()
# while True:
#     return_value, frame = cap.read()
#     if (return_value == False):
#         print("****************")
#         break
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     image = yolo.outcomes(image)  # 识别图片
#     result = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
#     curr_time = timer()
#     exec_time = curr_time - prev_time
#     prev_time = curr_time
#     accum_time = accum_time + exec_time
#     curr_fps = curr_fps + 1
#     if accum_time > 1:
#         accum_time = accum_time - 1
#         fps = "FPS: " + str(curr_fps)
#         curr_fps = 0
#     cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 0, 0), thickness=2)
#     cv2.namedWindow("result", cv2.WINDOW_NORMAL)
#     result = cv2.resize(result, (1000, 550), )
#     out.write(result)  # 保存识别之后的视频
#     cv2.imshow("result", result)  # 显示识别视频
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# out.release()
# cv2.destroyAllWindows()
# yolo.close_session()
