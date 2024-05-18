import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from uiPython.mainWindow import Ui_mainwindow
from uiPython.BucketTime import Ui_BucketTime

sys.path.append("/home/nuc2/PycharmProjects/yolov5-master")
import BucketTime_RD_vino
import multiprocessing
import cv2
import os
import logging
import signal
import serial.tools.list_ports as ser_list
import serial

_translate = QtCore.QCoreApplication.translate
redcode = -1
bluecode = 1
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')


def p1_start(code):
    global t_1
    # 切换川口的同时开始创建一个新的进程，在新的经常中调用32
    t_1 = multiprocessing.Process(target=track.main, args=(share_data, code))
    t_1.start()


# 利用一个控制器来控制页面的跳转
class Main_Controller:
    def __init__(self):
        self._mainWindow = Main_Window()
        self.Bucket = Bucket_Window()

    # 跳转到 mainWindow 窗口
    def mainWindow(self):
        self._mainWindow.switch_window_main.connect(self.BucketTime_to)
        self._mainWindow.show()

    # 跳转到 BucketTime_to 窗口, 注意关闭原页面
    def BucketTime_to(self):
        self.Bucket.switch_window_bucket.connect(self.mainWindow)
        self.Bucket.show()
        # 传入棋子的颜色
        self.Bucket.track_init()


class Main_Window(QWidget, Ui_mainwindow):
    switch_window_main = QtCore.pyqtSignal()  # 跳转信号

    def __init__(self):
        super(Main_Window, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.pushButton.clicked.connect(self.go_to_Bucket)
        self.pushButton.clicked.connect(self.close)
        self.pushButton_2.clicked.connect(self.close)
        # 获得传参参数中的棋子

    def go_to_Bucket(self):
        global code
        # 每一次切换都会检测到，选用哪个棋子
        code = bluecode if self.radioButton_2.isChecked() else redcode
        p1_start(code)
        self.switch_window_main.emit()


class Bucket_Window(QWidget, Ui_BucketTime):
    switch_window_bucket = QtCore.pyqtSignal()  # 跳转信号

    def __init__(self):
        super(Bucket_Window, self).__init__()
        self.timer_camera = QtCore.QTimer()  # 初始化定时器
        self.timer_camera.timeout.connect(self.show_viedo)
        self.setupUi(self)
        self.retranslateUi(self)
        # 定义按键
        self.logger = logging.getLogger("BucketTime")
        self.pushButton_4.clicked.connect(self.back_to_mainWindow)
        self.pushButton_3.clicked.connect(self.Restart)
        self.pushButton_4.clicked.connect(self.close)

    def track_init(self):
        self.timer_camera.start(1);

    def show_viedo(self):
        if share_data:
            try:
                Best_action, Better_action, Get_pos_ = None, None, None
                im0 = share_data['im0']
                img_reflect = share_data["Reflect_data"]["Reflect_img"]
                try:
                    Best_action = share_data["Reflect_data"]["Best"]
                    Better_action = share_data["Reflect_data"]["Better"]
                    Get_pos_ = True
                except Exception:
                    pass
                self.show_cv_img(im0, img_reflect, Best_action, Better_action, Get_pos_)
            except Exception:
                pass

    def show_cv_img(self, im0, img_reflect, reflect_best, reflect_better, Get_pos_):
        shrink_im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        shrink_reflect = cv2.cvtColor(img_reflect, cv2.COLOR_BGR2RGB)
        QtImg_im0 = QtGui.QImage(shrink_im0.data,
                                 shrink_im0.shape[1],
                                 shrink_im0.shape[0],
                                 shrink_im0.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)
        QtImg_reflect = QtGui.QImage(shrink_reflect.data,
                                     shrink_reflect.shape[1],
                                     shrink_reflect.shape[0],
                                     shrink_reflect.shape[1] * 3,
                                     QtGui.QImage.Format_RGB888)
        jpg_im0_out = QtGui.QPixmap(QtImg_im0).scaled(
            self.label_3.width(), self.label_3.height())
        jpg_Reflect_out = QtGui.QPixmap(QtImg_reflect).scaled(
            self.label_4.width(), self.label_4.height())
        self.label_3.setPixmap(jpg_im0_out)
        self.label_4.setPixmap(jpg_Reflect_out)
        self.label.setText(_translate("BucketTime", "Bucketing"))
        self.label_2.setText(_translate("BucketTime", f"Best {reflect_best}   Better {reflect_better}"))
        self.label_5.setText(_translate("BucketTime", f"Get_pos_:({Get_pos_})"))

    def back_to_mainWindow(self):
        """
        这是一个进程终止函数，由于子进程中的子进程设置了守护线程，因此再关闭这个新建进程时，所有的子进程也会一并关闭
        :return:
        """
        try:
            # 依次关闭所有进程
            print("close other process")
            for pid_ in share_data['pid']:
                os.kill(pid_, signal.SIGTERM)
            for ser_num in ser_list.comports():
                print(ser_num)
                ser2 = serial.Serial()
                ser2.baudrate = 115200  # 波特率
                ser2.port = ser_num
                ser2.close()
        except Exception:
            pass
        os.kill(t_1.pid, signal.SIGTERM)
        self.switch_window_bucket.emit()

    def Restart(self):
        try:
            print("close other process")
            for pid_ in share_data['pid']:
                os.kill(pid_, signal.SIGTERM)
            for ser_num in ser_list.comports():
                print(ser_num)
                ser2 = serial.Serial()
                ser2.baudrate = 115200  # 波特率
                ser2.port = ser_num
                ser2.close()
        except Exception:
            print("NO other Process")
        os.kill(t_1.pid, signal.SIGTERM)
        p1_start(code)
        self.track_init()


def main():
    global track, share_data
    share_data = multiprocessing.Manager().dict()
    track = BucketTime_RD_vino.Track()
    app = QtWidgets.QApplication(sys.argv)
    main_Controller = Main_Controller()  # 控制器实例
    main_Controller.mainWindow()  # 默认展示的是 hello 页面
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
