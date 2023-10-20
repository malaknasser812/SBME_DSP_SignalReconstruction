from scipy.fft import fft
import scipy.signal as sig
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
import pyqtgraph as pg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtWidgets, uic
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from numpy import *
import sys
import matplotlib

matplotlib.use('Qt5Agg')


class Sin_Signal():
    def __init__(self, name, mag, freq, phase):
        self.name = name
        self.mag = mag
        self.freq = freq 
        self.phase = phase
    
    


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi(r'Sampling_Studio.ui', self)
        
    
        self.pushButton_3.clicked.connect(lambda: self.displaySig())


        













def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()