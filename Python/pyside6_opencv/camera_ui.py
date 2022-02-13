# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'camera.ui'
##
## Created by: Qt User Interface Compiler version 6.2.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QLabel,
    QPushButton, QSizePolicy, QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(755, 500)
        self.lFrame = QLabel(Dialog)
        self.lFrame.setObjectName(u"lFrame")
        self.lFrame.setGeometry(QRect(10, 10, 640, 480))
        self.lFrame.setFrameShape(QFrame.Box)
        self.pbStart = QPushButton(Dialog)
        self.pbStart.setObjectName(u"pbStart")
        self.pbStart.setGeometry(QRect(660, 10, 89, 25))
        self.pbStop = QPushButton(Dialog)
        self.pbStop.setObjectName(u"pbStop")
        self.pbStop.setGeometry(QRect(660, 40, 89, 25))

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"OpenCV Capture", None))
        self.lFrame.setText(QCoreApplication.translate("Dialog", u"Captured Frame", None))
        self.pbStart.setText(QCoreApplication.translate("Dialog", u"Start", None))
        self.pbStop.setText(QCoreApplication.translate("Dialog", u"Stop", None))
    # retranslateUi

