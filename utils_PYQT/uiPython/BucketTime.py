# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BucketTime.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_BucketTime(object):
    def setupUi(self, BucketTime):
        BucketTime.setObjectName("BucketTime")
        BucketTime.resize(1308, 606)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(BucketTime.sizePolicy().hasHeightForWidth())
        BucketTime.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(BucketTime)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_4 = QtWidgets.QLabel(BucketTime)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setMinimumSize(QtCore.QSize(640, 480))
        self.label_4.setMaximumSize(QtCore.QSize(1440, 1080))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(36)
        font.setKerning(True)
        self.label_4.setFont(font)
        self.label_4.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_4.setLineWidth(5)
        self.label_4.setMidLineWidth(0)
        self.label_4.setScaledContents(False)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setWordWrap(False)
        self.label_4.setOpenExternalLinks(False)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(BucketTime)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setMinimumSize(QtCore.QSize(640, 480))
        self.label_3.setMaximumSize(QtCore.QSize(1440, 1080))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(72)
        self.label_3.setFont(font)
        self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_3.setLineWidth(5)
        self.label_3.setScaledContents(False)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(0, -1, -1, -1)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(BucketTime)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(26)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.label_5 = QtWidgets.QLabel(BucketTime)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(26)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout.addWidget(self.label_5)
        self.label_2 = QtWidgets.QLabel(BucketTime)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(26)
        self.label_2.setFont(font)
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAutoFillBackground(False)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton = QtWidgets.QPushButton(BucketTime)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(26)
        self.pushButton.setFont(font)
        self.pushButton.setMouseTracking(True)
        self.pushButton.setCheckable(False)
        self.pushButton.setAutoDefault(False)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(BucketTime)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(26)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setMouseTracking(True)
        self.pushButton_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_2.setCheckable(False)
        self.pushButton_2.setAutoDefault(False)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(BucketTime)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(26)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setMouseTracking(True)
        self.pushButton_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_3.setCheckable(False)
        self.pushButton_3.setAutoDefault(False)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_2.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(BucketTime)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(26)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setMouseTracking(True)
        self.pushButton_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_4.setCheckable(False)
        self.pushButton_4.setAutoDefault(False)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_2.addWidget(self.pushButton_4)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(BucketTime)
        QtCore.QMetaObject.connectSlotsByName(BucketTime)

    def retranslateUi(self, BucketTime):
        _translate = QtCore.QCoreApplication.translate
        BucketTime.setWindowTitle(_translate("BucketTime", "Form"))
        self.label_4.setText(_translate("BucketTime", "1034Robotics"))
        self.label_3.setText(_translate("BucketTime", "SZTU"))
        self.label.setText(_translate("BucketTime", "Loding...."))
        self.label_5.setText(_translate("BucketTime", "Loding..."))
        self.label_2.setText(_translate("BucketTime", "Loding...."))
        self.pushButton.setText(_translate("BucketTime", "Emergency stop"))
        self.pushButton_2.setText(_translate("BucketTime", " Stop"))
        self.pushButton_3.setText(_translate("BucketTime", "Restart"))
        self.pushButton_4.setText(_translate("BucketTime", "Quit"))