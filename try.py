from scipy.fft import fft
import scipy.signal as sig
import numpy as np
import pandas as pd
import pyqtgraph as pg
import os
import time
from scipy.interpolate import interp1d
import pyqtgraph as pg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtWidgets, uic
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from cmath import*
from numpy import *
import sys
import matplotlib

matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=7, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(1,1,1)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi(r'Sampling_Studio.ui', self)


        self.magnitude = 0
        self.frequency = 1
        self.phase = 0

        self.sinusoidal = 0
        self.signalSum = 0
        self.signal = 0
        self.signal_name = ""

        self.graph = pg.PlotItem() 

        self.canvas1 = MplCanvas(self, width=5, height=4, dpi=100)
        self.layout1 = QtWidgets.QVBoxLayout()
        self.layout1.addWidget(self.canvas1)

        self.canvas2 = MplCanvas(self, width=5, height=4, dpi=100)
        self.layout2 = QtWidgets.QVBoxLayout()
        self.layout2.addWidget(self.canvas2)

        self.canvas3 = MplCanvas(self.sampled_graph,  width=5, height=4, dpi=100)
        self.layout3 = QtWidgets.QVBoxLayout()
        self.layout3.addWidget(self.canvas3)


        #maping each signal with its variables
        self.signaldict = dict()
        self.signal_sum = 0 #sum of the added sin signals
        self.sin_signal_list = []


        # button connections
        self.showsignal_pushButton.clicked.connect(lambda: self.show_sin_signal())
        self.addtosum_pushButton.clicked.connect(lambda: self.display_summed_sin_signals())
        self.delete_signal_btn.clicked.connect(lambda: self.remove_sin_signal())
        self.send_sampler_btn.clicked.connect(lambda: self.send_to_sampler())


        self.time = arange(0.0, 1.0, 0.001)


    # define signal using given parameters ex: magnitude*sin(omega*self.time + theta)
    def signalParameters(self, magnitude, frequency, phase):
        omega = 2*pi*frequency
        theta = phase*pi/180
        return magnitude*sin(omega*self.time + theta)
    

    
    # get the required data for each the given signal
    def get_data (self):
        self.signal_name = self.sum_signals_combobox.currentText()
        self.indexList = self.signaldict[self.signal_name]
        self.signal = self.signalParameters(
            self.indexList[0], self.indexList[1], self.indexList[2])



    # plot the signal on the canvas created
    def plot_sin_signal(self, canvas, widget, layout, signal):
        canvas.axes.cla()
        canvas.axes.plot(self.time, signal)
        canvas.draw()
        widget.setCentralItem(self.graph)
        widget.setLayout(layout)



    # display the signal after specifying its properties
    def show_sin_signal(self):
        self.magnitude = float(self.mag_lineEdit.text())
        self.frequency = float(self.freq_lineEdit.text())
        self.phase = float(self.phase_lineEdit.text())
        self.name = (self.name_lineEdit.text())        
        if self.cos_radioButton.isChecked() == True: 
            #cosine wave will be drawn
            self.phase += 90

        self.signaldict[self.name] = self.magnitude, self.frequency, self.phase
        self.sinusoidal = self.signalParameters(
            self.magnitude, self.frequency, self.phase)
        self.plot_sin_signal(self.canvas1, self.show_signal_graph,
                        self.layout1, self.sinusoidal)
        



    # Add the sinusoidals generated
    def display_summed_sin_signals(self):
        self.sum_signals_combobox.addItem(self.name)
        self.signal_sum += self.sinusoidal
        self.plot_sin_signal(self.canvas2, self.summation_graph,
                        self.layout2, self.signal_sum)
        
        


    # remove selected signal
    def remove_sin_signal(self):
        if self.sum_signals_combobox.count() == 1:
            self.signal_sum = [0]*(len(self.time))
            self.signaldict.clear()
            self.sum_signals_combobox.clear()
        else:
            index = self.sum_signals_combobox.currentIndex()
            self.get_data()
            self.sum_signals_combobox.removeItem(index)
            self.signal_sum -= self.signal
            self.signaldict.pop(self.signal_name, None)
        self.plot_sin_signal(self.canvas2, self.summation_graph,
                        self.layout2, self.signal_sum)
        


    # send the signal to the sampler view
    def send_to_sampler (self):
        self.canvas3.axes.clear()
        self.x_data = self.time
        self.y_data = self.signal_sum
        
        self.plot_sin_signal(self.canvas3, self.sampled_graph, self.layout3, self.signal_sum)







def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()