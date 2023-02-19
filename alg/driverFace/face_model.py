import cv2
import numpy as np
import dlib
import os
import math


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def face_detect(filename, proc_id):
    model_points = np.array([
        [6.825897, 6.760612, 4.402142],
        [1.330353, 7.122144, 6.903745],
        [-1.330353, 7.122144, 6.903745],
        [-6.825897, 6.760612, 4.402142],
        [5.311432, 5.485328, 3.987654],
        [1.789930, 5.393625, 4.413414],
        [-1.789930, 5.393625, 4.413414],
        [-5.311432, 5.485328, 3.987654],
        [2.005628, 1.409845, 6.165652],
        [-2.005628, 1.409845, 6.165652],
        [2.774015, -2.080775, 5.048531],
        [-2.774015, -2.080775, 5.048531],
        [0.000000, -3.116408, 6.097667],
        [0.000000, -7.415691, 4.070434]])
    detector = dlib.get_frontal_face_detector()

    img0 = cv2.imread(filename)
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    faces = detector(img, 0)
    if faces:
        face = faces[0]
        x0, y0, x1, y1 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(img0, (x0, y0), (x1, y1), (255, 0, 0), 2)  # 画个人脸框框
        predictor = dlib.shape_predictor(
            'alg/driverFace/shape_predictor_68_face_landmarks.dat')
        ldmk = predictor(img, face)

        points = np.array([(p.x, p.y) for p in ldmk.parts()], dtype="double")

        points_68 = np.array([
            points[17],
            points[21],
            points[22],
            points[26],
            points[36],
            points[39],
            points[42],
            points[45],
            points[31],
            points[35],
            points[48],
            points[54],
            points[57],
            points[8]
        ], dtype="double")
        size = img.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, points_68, camera_matrix, dist_coeffs)

        theta = np.linalg.norm(rotation_vector)
        r = rotation_vector / theta
        R_ = np.array([[0, -r[2][0], r[1][0]],
                       [r[2][0], 0, -r[0][0]],
                       [-r[1][0], r[0][0], 0]])
        R = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * r * r.T + np.sin(theta) * R_
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        x = x * 180.0 / 3.141592653589793
        y = y * 180.0 / 3.141592653589793
        z = z * 180.0 / 3.141592653589793

        line = 'yaw:{:.1f}\npitch:{:.1f}\nroll:{:.1f}'.format(y, x, z)
        print('{},{}'.format(os.path.basename(filename), line.replace('\n', ',')))

        ylabel = 20
        for _, txt in enumerate(line.split('\n')):
            cv2.putText(img0, txt, (20, ylabel), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 1)
            ylabel = ylabel + 15

        for p in points_68:
            cv2.circle(img0, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1, 0)

        cv2.imwrite("alg/driverFace/output/face" + str(proc_id) + ".jpg", img0)

        if cv2.waitKey(-1) == 27:
            pass

        return np.array([y, x, z])
    else:
        return np.array([np.nan, np.nan, np.nan])


def driver_face_start_recog(img_path, equip_pos, proc_id):
    # 加载需要识别的图片
    # path = input("请输入图片路径：\n")
    x_y_z = face_detect(img_path, proc_id)
    res = -1
    if not np.isnan(x_y_z[0]):
        YAW = x_y_z[0]
        print('头部偏转方向：{}'.format(YAW))

        # 手动输入设备所在的位置，即设备在左边或者在右边，然后根据设备位置与头部偏转方向来判断：设备操作是否合规
        # position = input("输入设备所在的位置:")
        # left: equip_pos == 0
        if YAW >= 0 and equip_pos == 0:
            res = 0
            print("设备在左侧，操作合规")
        elif YAW <= 0 and equip_pos == 1:
            res = 0
            print("设备在右侧，操作合规")
        else:
            res = 1
            print("操作不合规")
    else:
        res = -1
        print('未检测到人脸')
    return res
