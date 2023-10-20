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
    def __init__(self):
        self.name = None
        self.mag = None
        self.freq = None 
        self.phase = None
        self.sinusoidal = 0
        self.malak 




        self.farah = 0
    
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi(r'Sampling_Studio.ui', self)

        #maping each signal with its variables
        self.signaldict = dict()
        self.showsignal_pushButton.clicked.connect(lambda: self.displaySig())

        # define signal using given parameters ex: magnitude*sin(omega*self.time + theta)
    def signalParameters(self, magnitude, frequency, phase):
        omega = 2*pi*frequency
        theta = phase*pi/180
        return magnitude*sin(omega*self.time + theta)

    def displaySig(self):
        signal1 =  Sin_Signal()
        signal1.mag = float(self.mag_lineEdit.text())
        signal1.freq = float(self.freq_lineEdit.text())
        signal1.phase = float(self.phase_lineEdit.text())
        signal1.name = (self.name_lineEdit.text())

        
        if self.cos_radioButton.isChecked() == True: 
            #cosine wave will be drawn
            self.phase += 90

        self.signaldict[signal1.name] = signal1.mag, signal1.freq, signal1.phase
        self.comboBox_4.addItem(signal1.name)
        #computes the representation of the sin wave
        signal1.sinusoidal = self.signalParameters(signal1.mag, signal1.freq, signal1.phase) 
        self.signalPlot(self.canvas1, self.sinoPlotter,
                        self.layout1, self.sinusoidal)






        













def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()