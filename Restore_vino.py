"""
brief :Editor cjh
"""
import argparse
import math
import time

import sys
import cv2
import matplotlib

matplotlib.use("Qt5Agg")  # 声明使用pyqt5

import numpy as np
import serial
from Map_Reflect_utils import Realsense
from utils_track import vino as ov
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from utils_PYQT.uiPython.TrackerTime import Ui_TrackerTime
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')

pi = math.pi
_translate = QtCore.QCoreApplication.translate


def open_ser() -> bool:
    global ser2
    port0 = 'COM13'
    ser2 = serial.Serial()
    ser2.baudrate = 115200  # 波特率
    try:
        ser2.port = port0
        ser2.open()
        if ser2.isOpen() == True:
            print("串口打开成功")
        return True
    except Exception:
        return False
        pass


def send_msg(jd):
    try:
        ser2.write(str(jd).encode("gbk"))
    except Exception as exc:
        print("发送异常", exc)


def send_data_change(data: float) -> str:
    data = round(data)
    data_postive = abs(data)
    if data > 0:
        data_ = str("P")
    else:
        data_ = str("N")
    if data_postive >= 1000:
        data_ += str(data_postive)
    elif data_postive >= 100:
        data_ += str("0") + str(data_postive)
    elif data_postive >= 10:
        data_ += str("00") + str(data_postive)
    elif data_postive >= 0:
        data_ += str("000") + str(data_postive)
    return data_


def data_solve(best_coord, dist_):
    # 计算x，y的点和距离
    str_send = str('kk') + send_data_change(best_coord[0] * 1000) + send_data_change(
        best_coord[1] * 1000) + send_data_change(dist_ * 10)
    print(str_send)
    send_msg(str_send)





class track(object):
    def __init__(self):
        super(track, self).__init__()
        self.realsense_cap = Realsense.realsense()
        self.depth_intrin = None
        self.aligned_depth_frame = None

    def laser(self):
        # todo 为了纠正imu的累积误差
        pass

    # 计算最短距离
    def calc(self, x, y, center):
        return math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # 修改图像的对比度,coefficent>0, <1降低对比度；>1提升对比度 建议0-2
    def change_contrast(self, img, coefficent):
        imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m = cv2.mean(img)[0]
        graynew = m + coefficent * (imggray - m)
        img1 = np.zeros(img.shape, np.float32)
        k = np.divide(graynew, imggray, out=np.zeros_like(graynew), where=imggray != 0)
        img1[:, :, 0] = img[:, :, 0] * k
        img1[:, :, 1] = img[:, :, 1] * k
        img1[:, :, 2] = img[:, :, 2] * k
        img1[img1 > 255] = 255
        img1[img1 < 0] = 0
        return img1.astype(np.uint8)

    # 修改图像的亮度，brightness取值0～2 <1表示变暗 >1表示变亮
    def change_brightness(self, img, brightness):
        [averB, averG, averR] = np.array(cv2.mean(img))[:-1] / 3
        k = np.ones(img.shape)
        k[:, :, 0] *= averB
        k[:, :, 1] *= averG
        k[:, :, 2] *= averR
        img = img + (brightness - 1) * k
        img[img > 255] = 255
        img[img < 0] = 0
        return img.astype(np.uint8)

    def image_stream(self):
        align = self.realsense_cap.cam_init(640)  # 引用相机,初始化相机
        return align  # 返回时，退出函数

    def camera_to_world(self, xc, yc, zc):
        # xc,yc,zc是相机坐标系下的坐标，现在转化为世界坐标系下的坐标
        RT = self.pose_robot(0, 120, 0, 0, 0, 0.30)
        trans = np.array([xc, yc, zc, 1]).T
        x_tran, y_tran, z_tran, _ = (RT @ trans)  # 传出作为全局变量s
        return x_tran, y_tran, z_tran

    def myRPY2R_robot(self, x, y, z):
        Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return R

    # 用于根据位姿计算变换矩阵
    def pose_robot(self, x, y, z, Tx, Ty, Tz):
        thetaX = x / 180 * pi
        thetaY = y / 180 * pi
        thetaZ = z / 180 * pi
        R = self.myRPY2R_robot(thetaX, thetaY, thetaZ)
        t = np.array([[Tx], [Ty], [Tz]])
        RT1 = np.column_stack([R, t])  # 列合并
        RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
        # RT1=np.linalg.inv(RT1)
        return RT1

    def run_track(self, model, cap, ser_flag):
        start = time.time()
        min_dist = 1000
        frame, image_depth, self.depth_intrin, self.aligned_depth_frame = self.realsense_cap.cam_run(cap)
        frame, det = model.run(frame)
        # 初始化几个值,面对没有出现目标的时候
        coord = (int(frame.shape[1] / 2), frame.shape[0])
        coord_3d = (0, 0, 0)
        dist_last = 0
        if len(det):
            for xyxy, conf, cls in reversed(det):  # 识别物的类别像素坐标，置信度，物体类别号
                mid_pos = [int((xyxy[0] + xyxy[2]) / 2),
                           int((xyxy[1] + xyxy[3]) / 2)]
                cv2.circle(frame, mid_pos, 2, (0, 255, 0), 2)
                min_val = min(abs(int(xyxy[2]) - int(xyxy[0])), abs(int(xyxy[3]) - int(xyxy[1])))
                # 将中心点向四周扩展计算深度
                z, y, x = self.realsense_cap.depth_to_data(self.depth_intrin, self.aligned_depth_frame, mid_pos,
                                                           min_val)
                # 换到车上坐标系
                x, y, z = self.camera_to_world(x, y, z)
                x, y, z = round(x, 2), round(y, 2), round(z, 2)
                # 添加目标物参数
                dist_last = self.calc(x, y, (0, 0))
                if min_dist > dist_last:
                    min_dist = dist_last
                    coord = mid_pos
                    coord_3d = (x, y, z)
        if ser_flag:
            data_solve(coord_3d, round(dist_last * 100, 2))
        end = time.time()
        fps = 1 / (end - start)
        cv2.line(frame, (int(frame.shape[1] / 2), frame.shape[0]), coord, (255, 0, 0), 3)
        cv2.putText(frame, f"{round(fps, 1)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        return frame, min_dist, coord_3d, coord

    # This function is used to choose argument,which is called init function
    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_path', type=str, default="/home/nuc2/PycharmProjects/yolov5-master/best_ball.xml",
                            help='model path')
        parser.add_argument('--weights_path', nargs='+', type=str, default="/home/nuc2/PycharmProjects/yolov5-master"
                                                                           "/best_ball.bin",
                            help='weights path or triton URL')
        parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold')
        parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--iou-thres', type=float, default=0.55, help='NMS IoU threshold')
        parser.add_argument('--device', default='GPU', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--name-porcess', default='Restore', help='name to processs--cjh')

        opt = parser.parse_args()

        return opt

    # start function
    def track_init(self):
        # 使其为全局变量，不然无法添加
        opt = self.parse_opt()
        vino_track = ov.Vino(**vars(opt))
        cap = self.image_stream()
        ser_flag = open_ser()
        return vino_track, cap, ser_flag


class Track_Window(QWidget, Ui_TrackerTime):
    def __init__(self):
        super(Track_Window, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        # 实例化对象
        self.track_time = track()
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_cv_img)
        self.logger = logging.getLogger("TrackTime")
        # 定义按键
        # start
        self.pushButton.clicked.connect(self.track_init)
        # Restart
        self.pushButton_3.clicked.connect(self.Restart)
        # stop
        self.pushButton_4.clicked.connect(self.Close)

    def plotcos(self):
        t = np.arange(0.0, 5.0, 5)
        s = np.cos(2 * np.pi * t)
        self.canvas.set_mat_func(t, s)
        self.canvas.plot_tick()

    def track_init(self):
        global model, cap, ser_flag
        model, cap, ser_flag = self.track_time.track_init()
        self.timer_camera.start(1);

    def Restart(self):
        try:
            self.track_time.realsense_cap.pipeline.stop()
        except:
            print("No this Process")
        self.logger.warning("Restart")
        self.track_init()

    def Close(self):
        self.timer_camera.stop()
        try:
            # 关闭视频流
            self.track_time.realsense_cap.pipeline.stop()
        except:
            print("No this Process")
        self.logger.warning("Close")
        self.close()

    def show_cv_img(self):
        frame, min_dist, coord_3d, mid_pos = self.track_time.run_track(model, cap, ser_flag)
        shrink_im0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        QtImg_im0 = QtGui.QImage(shrink_im0.data,
                                 shrink_im0.shape[1],
                                 shrink_im0.shape[0],
                                 shrink_im0.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)
        jpg_im0_out = QtGui.QPixmap(QtImg_im0).scaled(
            self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(jpg_im0_out)
        self.label.setText(_translate("TrackerTime", f"mid_pos:{mid_pos}"))
        self.label_5.setText(_translate("TrackerTime", f"min_dist {round(min_dist, 2)}"))
        self.label_2.setText(_translate("TrackerTime", f"coord_3d:({coord_3d})"))


def main():
    app = QtWidgets.QApplication(sys.argv)
    Track_time = Track_Window()
    Track_time.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

    # track.main()
