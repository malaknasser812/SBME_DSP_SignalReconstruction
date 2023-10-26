from scipy.fft import fft
import scipy.signal as sig
from scipy import interpolate
import numpy as np
import pandas as pd
import pyqtgraph as pg
import os
import time
from scipy.interpolate import interp1d
import pyqtgraph as pg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene ,QLabel , QHBoxLayout
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
        self.noise_flag = False

        self.sinusoidal = 0
        self.signalSum = 0
        self.signal = 0
        self.signal_name = ""

        self.loaded = False
        self.graph = pg.PlotItem() 
        self.maxFreq =0
        self.normFreq_index = 0
        self.y_data = []
        self.x_data = []
        self.y_noisy = []

        self.canvas1 = MplCanvas(self, width=5, height=4, dpi=100)
        self.layout1 = QtWidgets.QVBoxLayout()
        self.layout1.addWidget(self.canvas1)

        self.canvas2 = MplCanvas(self, width=5, height=4, dpi=100)
        self.layout2 = QtWidgets.QVBoxLayout()
        self.layout2.addWidget(self.canvas2)

        self.canvas3 = MplCanvas(self.sampled_graph,  width=5, height=4, dpi=100)
        self.layout3 = QtWidgets.QVBoxLayout()
        self.layout3.addWidget(self.canvas3)

        self.canvas4 = MplCanvas(self.recovered_graph,  width=5, height=4, dpi=100)
        self.layout4 = QtWidgets.QVBoxLayout()
        self.layout4.addWidget(self.canvas4)
        
        self.canvas5 = MplCanvas(self.error_graph,  width=5, height=4, dpi=100)
        self.layout5 = QtWidgets.QVBoxLayout()
        self.layout5.addWidget(self.canvas5)

        #maping each signal with its variables
        #store signal names as keys and corresponding signal parameters (magnitude, frequency, phase) as values.
        self.signaldict = dict()
        self.signal_sum = 0 #sum of the added sin signals
        self.sin_signal_list = []

        #setting the min and max values of the SNR value
        self.SNR_slider.setMinimum(2)
        self.SNR_slider.setMaximum(30)
        self.SNR_slider.setValue(2)


        #lineEdits connections
        self.name_lineEdit.returnPressed.connect(self.name_lineEdit_returnPressed)
        self.mag_lineEdit.returnPressed.connect(self.mag_lineEdit_returnPressed)
        self.freq_lineEdit.returnPressed.connect(self.freq_lineEdit_returnPressed)

        # button connections
        self.showsignal_pushButton.clicked.connect(lambda: self.show_sin_signal())
        self.addtosum_pushButton.clicked.connect(lambda: self.display_summed_sin_signals())
        self.delete_signal_btn.clicked.connect(lambda: self.remove_sin_signal())
        self.send_sampler_btn.clicked.connect(lambda: self.send_to_sampler())
        self.load_btn.clicked.connect(lambda: self.load())
        self.sample_rate_comboBox.addItem("Normalized Frequency")
        self.sample_rate_comboBox.addItem("Actual Frequency")
        self.sample_rate_comboBox.activated.connect(self.update_slider_labels)
        self.freq_slider.valueChanged.connect(lambda: self.plotHSlide())
        self.freq_slider.valueChanged.connect(self.update_slider_labels)
        self.SNR_slider.valueChanged.connect(self.SNR_value_change)
        self.add_noise_checkbox.stateChanged.connect(lambda : self.toggle_noise())
        self.time = arange(0.0, 2.0, 0.001)


    #make enter-click work to line edit 
    def name_lineEdit_returnPressed(self):
        self.mag_lineEdit.setFocus()
    def mag_lineEdit_returnPressed(self):
        self.freq_lineEdit.setFocus()
    def freq_lineEdit_returnPressed(self):
        self.phase_lineEdit.setFocus()
    # define signal using given parameters ex: magnitude*sin(omega*self.time + theta)
    def signalParameters(self, magnitude, frequency, phase):
        omega = 2*pi*frequency
        theta = phase*pi/180
        return magnitude*sin(omega*self.time + theta)
    

    
    # get the required data for each selected signal
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
        self.signaldict[self.name] = self.magnitude, self.frequency, self.phase
        self.sinusoidal = self.signalParameters(
            self.magnitude, self.frequency, self.phase)
        self.plot_sin_signal(self.canvas1, self.show_signal_graph,
                        self.layout1, self.sinusoidal)
        



    # Add the sinusoidals generated
    def display_summed_sin_signals(self):
        #temp_sum = self.signal_sum + self.sinusoidal
        #self.sum_signals_combobox.addItem(self.name)
        # Check if temp_sum is empty (all zeros)
        #if any(temp_sum):
            #self.signal_sum = temp_sum
        #else:
            # If temp_sum is empty, initialize self.signal_sum as a NumPy array
            #self.signal_sum = np.zeros(2000)
            
        self.signal_sum += self.sinusoidal
        self.sum_signals_combobox.addItem(self.name)
        self.plot_sin_signal(self.canvas2, self.summation_graph,
                        self.layout2, self.signal_sum)
        


    # remove selected signal
    def remove_sin_signal(self):
        if self.sum_signals_combobox.count() == 0:
            warning = QMessageBox()
            warning.setIcon(QMessageBox.Warning)
            warning.setText("There are no more signals to delete.")
            warning.exec_()
        elif self.sum_signals_combobox.count() == 1:
            self.signal_sum = []  # Empty list
            self.signaldict.clear()
            self.sum_signals_combobox.clear()
            
        else:
            #Removing signals when there's more than one signal
            index = self.sum_signals_combobox.currentIndex()
            self.get_data()
            self.sum_signals_combobox.removeItem(index)
            self.signal_sum -= self.signal
            self.signaldict.pop(self.signal_name, None)
        
        # Only plot if there are signals in self.signal_sum
        if any(self.signal_sum):
            self.plot_sin_signal(self.canvas2, self.summation_graph, self.layout2, self.signal_sum)
        else:
            # Clear the graph if there are no signals
            self.canvas2.axes.clear()
            self.layout2.removeWidget(self.canvas2)
        self.canvas2.draw()
        
            #NOISE by farooo7aaaaaa
   
        


    # send the signal to the sampler view
    def send_to_sampler (self):
        self.canvas3.axes.clear()
        self.canvas4.axes.clear()
        self.canvas5.axes.clear()
        self.x_data = self.time
        self.y_data = self.signal_sum
        self.maxFreq = self.get_fmax()
        self.main_layout.setCurrentWidget(self.sampler_tab)
        self.plot_graph(self.signal_sum)



    def load(self):
        # Clear the plots in canvas3 , canvas4 and canvas5
        self.canvas3.axes.clear()
        self.canvas4.axes.clear()
        self.canvas5.axes.clear()

        self.fname1 = QFileDialog.getOpenFileName(
            None, "Select a file...", os.getenv('HOME'), filter="All files (*)")
        path1 = self.fname1[0]
        data1 = pd.read_csv(path1)
        # Extract signal data (Y) and time data (X) from the CSV file
        self.y_data = data1.values[:, 1]
        self.x_data = data1.values[:, 0]
        # Set the value of the horizontal slider initially to its minimum value
        self.freq_slider.setValue(self.freq_slider.minimum())
        # If data is already loaded, remove canvas3 and canvas4 from their layouts
        if self.loaded == True:
            self.layout3.removeWidget(self.canvas3)
            self.layout4.removeWidget(self.canvas4)
            self.layout5.removeWidget(self.canvas5)
        self.maxFreq = self.get_fmax()
        self.loaded = True
        self.plot_graph(self.y_data)

    
    def get_fmax(self):
        FTydata = np.fft.fft(self.y_data)
        # Keep only the first half of the FFT data (positive frequencies)
        FTydata = FTydata[0:int(len(self.y_data)/2)]
        FTydata = abs(FTydata)
        # Find the maximum amplitude in the FFT data
        maxamp = max(FTydata)
        #A noise threshold is defined as 1% of the maximum amp. used to identify significant frequency components above the noise level.
        noise = (maxamp/100)
        #finds the indices where the FFT values are significant(more than noise threshold) and not considered noise.
        self.fmaxtuble = np.where(FTydata > noise)
        #finds the index with the maximum value which represents the dominant frequency component in the signal
        self.maxFreq = max(self.fmaxtuble[0])
        return self.maxFreq

    def sinc_interp(self, sample_data,sample_time , original_time):

            #It's important that the signal values and the corresponding time values have matching lengths for interpolation to be meaningful 
            if len(sample_data) != len(sample_time):
                raise ValueError('sample_data and sample_time must be the same length')
            
            # Check if sample_time has more than one element before computing T
            if len(sample_time) > 1:
                T = sample_time[1] - sample_time[0]
            else:
                T = 1 
            # converting to 2D array In signal processing and interpolation, 
            # working with 2D arrays (matrices) often allows for more efficient and vectorized computations
            sincM = np.tile(original_time, (len(sample_time), 1)) - np.tile(sample_time[:, np.newaxis], (1, len(original_time)))
            #calculates a weighted sum of the resampled data x using the sinc function values
            interpolated_data = np.dot(sample_data, np.sinc(sincM/T))
            return interpolated_data
    # make a function for plotting the noise and 
    # if value of
    def plot_graph(self, y_data):           

            selected_option = self.sample_rate_comboBox.currentIndex()
            #choosing normalized freq. so dependently of fmax
            if selected_option == 0 :
                self.freq_slider.setMaximum(int(4*self.maxFreq))
            else: #actual freq.
                self.freq_slider.setMaximum(60)

            # smapling the data and stored in variable contains both the resampled signal and its associated time values.
            sample_data, _ = sig.resample(y_data, self.freq_slider.value(), self.x_data) 
            # Ensure that sample_data and sample_time have the same length
            sample_time = np.linspace(self.x_data[0], self.x_data[-1], len(sample_data))

            
            # Perform interpolation to estimate the sampled data
            f = interpolate.interp1d(self.x_data, y_data, kind='linear')
            sample_data = f(sample_time)
            #interpolatng on the new data 
            recontructed_data = self.sinc_interp(sample_data, sample_time, self.x_data)
            # Calculate the error between the original signal and the reconstructed signal
            error = y_data - recontructed_data
            # plotting the original signal and the sampled data as dots 
            self.canvas3.axes.plot(self.x_data, y_data,color='b')
            self.canvas3.axes.scatter(sample_time, sample_data, color='k', s=10)
            # self.canvas4.axes.legend(labels=self.signal_name)  # Add a legend
            self.canvas3.draw()
            self.sampled_graph.setCentralItem(self.graph)
            self.sampled_graph.setLayout(self.layout3)
            # plotting the constructed data on the second graph
            self.canvas4.axes.plot(self.x_data, recontructed_data, color='r')
            self.canvas4.draw()
            self.recovered_graph.setCentralItem(self.graph)
            self.recovered_graph.setLayout(self.layout4)
            # plotting the error difference between 2 graphs in 3rd graph
            self.canvas5.axes.plot(self.x_data, error, color='g')
            self.canvas5.draw()
            self.error_graph.setCentralItem(self.graph)
            self.error_graph.setLayout(self.layout5)
        


    def plotHSlide(self):
        self.canvas3.axes.clear()
        self.canvas4.axes.clear()
        self.canvas5.axes.clear()
        if self.noise_flag:
            self.plot_graph(self.y_noisy)
        else:
            self.plot_graph(self.y_data)


    def update_slider_labels(self,value):
            # current_value = self.freq_slider.value()
            selected_option = self.sample_rate_comboBox.currentIndex()
            self.freq_slider.setMinimum(1)  # Set the minimum value of both cases
            if selected_option == 0 : # 0 corresponds to "Normalized Frequency"
                self.freq_slider.setMaximum(int(4 * self.maxFreq))  # Set the maximum value in case 1       
                self.sliderlabel.setText(f'Fmax= {self.maxFreq / 2}Hz <br>{value//self.maxFreq}Fmax')
                self.currentvalue.setText(f'CurrentValue = {value}Hz')
                # self.sliderlabel.setText(f'Fmax={self.maxFreq}Hz')
            else:
                self.freq_slider.setMaximum(60)
                self.sliderlabel.setText(f'{value} Hz')
                self.currentvalue.setText(f'CurrentValue = {value}Hz')

    
    def toggle_noise(self):
        #toggle the noise Flag -> whether to add noise or not
        self.noise_flag = not self.noise_flag 
        self.add_gaussian_noise(self.y_data)

    #When the SNR slider changes value
    def SNR_value_change(self,):
        self.add_gaussian_noise(self.y_data)
    

    #Adding Noise to the signal        
    def add_gaussian_noise(self,y_data):
        #if the noise checkbox is true -> add noise to the signal
        if self.noise_flag:
            # Calculate the power of the signal -> computes the mean of the squared values of y_data.
            signal_power = np.mean(np.square(y_data))

            # Calculate the desired noise power based on SNR
            noise_power = signal_power / (10**(self.SNR_slider.value() / 10))

            # Generate white Gaussian noise
            noise = np.random.normal(0, np.sqrt(noise_power), len(y_data))

            # Add noise to the signal
            y_noisy = y_data + noise
            
            # Plotting the new noisy signal 
            self.canvas3.axes.cla()
            self.canvas4.axes.cla()
            self.canvas5.axes.cla()
            self.y_noisy=y_noisy
            self.plot_graph(y_noisy)

        # else draw the original data given (without noise)
        else:
            self.canvas3.axes.cla()
            self.canvas4.axes.cla()
            self.canvas5.axes.cla()
            self.y = y_data
            self.plot_graph(self.y_data)

    
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()