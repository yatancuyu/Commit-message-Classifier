from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from models import get_results


class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data[index.row(), index.column()]
            return str(value)[:5]

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(['tfidf', 'tf', 'bin'][section])

            if orientation == Qt.Vertical:
                return str(['nothing', 'stemming', 'lemmatization'][section])


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(817, 484)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 10, 251, 51))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(40, 80, 161, 21))
        self.radioButton.setObjectName("radioButton")
        self.buttonGroup_2 = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup_2.setObjectName("buttonGroup_2")
        self.buttonGroup_2.addButton(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(40, 110, 141, 23))
        self.radioButton_2.setObjectName("radioButton_2")
        self.buttonGroup_2.addButton(self.radioButton_2)
        self.radioButton_3 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_3.setGeometry(QtCore.QRect(40, 140, 112, 23))
        self.radioButton_3.setObjectName("radioButton_3")
        self.buttonGroup_2.addButton(self.radioButton_3)
        self.radioButton_4 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_4.setGeometry(QtCore.QRect(40, 170, 112, 23))
        self.radioButton_4.setObjectName("radioButton_4")
        self.buttonGroup_2.addButton(self.radioButton_4)
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(270, 60, 403, 111))
        self.tableView.setObjectName("tableView")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 200, 211, 25))
        self.pushButton.setObjectName("pushButton")
        self.buttonGroup_2.addButton(self.pushButton)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(240, 20, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 817, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Choose classifier"))
        self.radioButton.setText(_translate("MainWindow", "LogisticRegression"))
        self.radioButton_2.setText(_translate("MainWindow", "DecisionTree"))
        self.radioButton_3.setText(_translate("MainWindow", "NaiveBayes"))
        self.radioButton_4.setText(_translate("MainWindow", "SVC"))
        self.pushButton.setText(_translate("MainWindow", "Train and Get F1 score"))
        self.label_2.setText(_translate("MainWindow", "Results"))

        self.pushButton.clicked.connect(self.btn_clk)

    def btn_clk(self):
        radio_btns_status = [self.radioButton.isChecked(), self.radioButton_2.isChecked(),
                             self.radioButton_3.isChecked(), self.radioButton_4.isChecked()]
        # default=logreg
        if not any(radio_btns_status):
            radio_btns_status[0] = True

        models = ['logreg', 'decision_tree', 'bayes', 'svc']
        model = models[radio_btns_status.index(True)]

        results = get_results(model)

        table = TableModel(results)
        self.tableView.setModel(table)
