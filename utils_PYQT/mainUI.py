"""
brief :Editor cjh
"""
import sys

sys.path.append("/home/nuc2/PycharmProjects/yolov5-master")
import BucketTime_Reflect_vino

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from uiPython.mainWindow import Ui_mainwindow
from uiPython.BucketTime import Ui_BucketTime
import multiprocessing
import cv2
import sys
import os
import signal

bluecode = 1.0
redcode = -1.0

_translate = QtCore.QCoreApplication.translate


def p1_start(code):
    global t_1
    # 切换川口的同时开始创建一个新的进程，在新的经常中调用32
    t_1 = multiprocessing.Process(target=track.main, args=(share_data,))
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
        self.Bucket.track_init()


def exit_() -> None:
    """
    exit all process
    :return:
    """
    try:
        sys.exit(0)
    except:
        print('die')
    finally:
        print('cleanup')


class Main_Window(QWidget, Ui_mainwindow):
    switch_window_main = QtCore.pyqtSignal()  # 跳转信号

    def __init__(self):
        super(Main_Window, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.pushButton.clicked.connect(self.go_to_Bucket)
        self.pushButton.clicked.connect(self.close)

    def go_to_Bucket(self):
        # bule 打开
        global code
        if self.radioButton.isChecked():
            code = bluecode
        # red 打开
        elif self.radioButton_2.isChecked():
            code = redcode
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
        self.pushButton_4.clicked.connect(self.back_to_mainWindow)
        self.pushButton_3.clicked.connect(self.Restart)
        self.pushButton_4.clicked.connect(self.close)

    def track_init(self):
        self.timer_camera.start(1);

    def show_viedo(self):
        if share_data:
            try:
                Best_action, Better_action, Best_x = None, None, None
                im0 = share_data['im0']
                img_reflect = share_data["Reflect_data"]["Reflect_img"]
                try:
                    Best_action = share_data["Reflect_data"]["Best"]
                    Better_action = share_data["Reflect_data"]["Better"]
                    Best_x = share_data['Best_x']
                except Exception:
                    pass
                self.show_cv_img(im0, img_reflect, Best_action, Better_action, Best_x)
            except Exception:
                pass

    def show_cv_img(self, im0, img_reflect, Best_action, Better_action, Best_x):
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
        self.label_2.setText(_translate("BucketTime", f"Best {Best_action}   Better {Better_action}"))
        self.label_5.setText(_translate("BucketTime", f"Track:({Best_x})"))

    def back_to_mainWindow(self):
        """
        这是一个进程终止函数，由于子进程中的子进程设置了守护线程，因此再关闭这个新建进程时，所有的子进程也会一并关闭
        :return:
        """
        try:
            # 依次关闭所有进程
            print("close other process")
            for pid_ in share_data['pid']:
                print(pid_)
                os.kill(pid_, signal.SIGTERM)
        except Exception:
            pass
        os.kill(t_1.pid, signal.SIGTERM)
        self.switch_window_bucket.emit()

    def Restart(self):
        try:
            print("close other process")
            for pid_ in share_data['pid']:
                print(pid_)
                os.kill(pid_, signal.SIGTERM)
        except Exception:
            print("NO other Process")
        os.kill(t_1.pid, signal.SIGTERM)
        p1_start(code)
        self.track_init()


def main():
    # sys.exit()
    global track, share_data
    share_data = multiprocessing.Manager().dict()
    track = BucketTime_Reflect_vino.Track()
    app = QtWidgets.QApplication(sys.argv)
    main_Controller = Main_Controller()  # 控制器实例
    main_Controller.mainWindow()  # 默认展示的是 hello 页面
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
