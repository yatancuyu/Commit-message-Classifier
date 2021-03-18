import sys
from PyQt5.QtWidgets import QMainWindow
from qtpy import QtWidgets
import warnings
from MainWindow import Ui_MainWindow

warnings.simplefilter("ignore")


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = MainWindow()
    widget.resize(700, 300)
    widget.setWindowTitle("Commit Classifier")
    widget.show()
    exit(app.exec_())
