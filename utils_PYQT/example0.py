from p1 import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
import cv2


class UiMain(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(UiMain, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.loadImage)
        self.frame = None

    # 打开文件功能
    def loadImage(self):
        self.fname = QFileDialog.getOpenFileName(self, '请选择图片', '.', '图像文件(*.jpg *.jpeg *.png)')
        if self.fname:
            jpg = QtGui.QPixmap(self.fname[0]).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(jpg)


        else:
            print("fail to load!!")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = UiMain()
    ui.show()
    sys.exit(app.exec_())
