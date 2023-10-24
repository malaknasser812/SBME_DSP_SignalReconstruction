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
        self.SNR_LVL = 1

        self.sinusoidal = 0

        self.signal_name = ""

        self.loaded = False
        self.graph = pg.PlotItem() 
        self.maxFreq =0
        self.normFreq_index = 0
        self.y_data = []
        self.x_data = []

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
        self.SNR_slider.setMinimum(0)
        self.SNR_slider.setMaximum(30)
        self.SNR_slider.setValue(self.SNR_slider.maximum())

        # button connections
        self.showsignal_pushButton.clicked.connect(lambda: self.show_sin_signal())
        self.addtosum_pushButton.clicked.connect(lambda: self.display_summed_sin_signals())
        self.delete_signal_btn.clicked.connect(lambda: self.remove_sin_signal())
        self.send_sampler_btn.clicked.connect(lambda: self.send_to_sampler())
        self.load_btn.clicked.connect(lambda: self.load())
        self.sample_rate_comboBox.addItem("Normalized Frequency")
        self.sample_rate_comboBox.addItem("Actual Frequency")
        self.sample_rate_comboBox.activated.connect(lambda:self.plot(self.y_data))
        self.sample_rate_comboBox.currentIndexChanged.connect(self.update_slider_labels)
        self.freq_slider.valueChanged.connect(lambda: self.plotHSlide())
        #self.add_noise_checkbox.stateChanged.connect(lambda : self.toggle_noise)
        #self.SNR_slider.valueChanged.connect(lambda: self.SNR_value_change)

        self.time = arange(0.0, 2.0, 0.001)


    # define signal using given parameters ex: magnitude*sin(omega*self.time + theta)
    def signalParameters(self, magnitude, frequency, phase):
        omega = 2*pi*frequency
        theta = phase*pi/180
        return magnitude*sin(omega*self.time + theta)
    

    
    # get the required data for each selected signal
    def get_data (self):
        self.signal_name = self.sum_signals_combobox.currentText()
        if self.signal_name is not None:
            # Check if self.signal_name exists in the dictionary
            if self.signal_name in self.signaldict:
                self.indexList = self.signaldict[self.signal_name]
                self.signal = self.signalParameters(
                    self.indexList[0], self.indexList[1], self.indexList[2])
        else:
            # Handle the case where the signal_name doesn't exist in the dictionary
            raise ValueError('No more signals to display')
        



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
        

    #NOISE by farooo7aaaaaa
    def update_noise_level(self):
        noise_level = self.SNR_slider.value()  # Get the current noise level from the slider
        self.add_noise_based_on_snr(noise_level)
    

    def add_noise_based_on_snr(self, snr_level):
        # Calculate the signal power (you may need to adjust this based on your signal)
        signal_power = np.mean(np.square(self.sinusoidal))

        # Calculate the desired noise power based on the SNR level
        noise_power = signal_power / (10**(snr_level / 10))

        # Generate Gaussian noise with the specified noise power
        noise = np.random.normal(0, np.sqrt(noise_power), len(self.time))

        # Add the noise to the signal
        noisy_signal = self.sinusoidal + noise

        # Plot the noisy signal
        self.plot_sin_signal(self.canvas3, self.sampled_graph, self.layout3, noisy_signal)
        
        
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

        


    # send the signal to the sampler view
    def send_to_sampler (self):
        self.canvas3.axes.clear()
        self.x_data = self.time
        self.y_data = self.signal_sum
        max_freq = []
        for i in self.signaldict: 
            max_freq.append(self.signaldict[i][1])
        self.maxFreq = max(max_freq)
        self.plot_sin_signal(self.canvas3, self.sampled_graph, self.layout3, self.signal_sum)
        self.main_layout.setCurrentWidget(self.sampler_tab)



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

        # Filtering
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
        print(self.maxFreq)
        self.loaded = True
        self.plot(self.y_data)
    

    def plot(self, y_data):
        selected_option = self.sample_rate_comboBox.currentIndex()
        #choosing normalized freq. so dependently of fmax
        if selected_option == self.normFreq_index:
            self.freq_slider.setMaximum(int(ceil(4*self.maxFreq)))
        else: #actual freq.
            self.freq_slider.setMaximum(60)

        # sampling the data and stored in variable contains both the resampled signal and its associated time values.
        resample_data = sig.resample(y_data, self.freq_slider.value(), self.x_data)

        sample_data = resample_data[0]
        sample_time = resample_data[1]
        # ensure that the first sample has the same time and value as the original data and that the last sample also matches the original data
        if len(sample_time) > 0:
            sample_time[0]=self.x_data[0]
        sample_data[0]=y_data[0]
        sample_time=np.append(sample_time,[self.x_data[-1]])
        sample_data=np.append(sample_data,[y_data[-1]])

        #interpolatng on the new data 
        recontructed_data = self.sinc_interp(sample_data, sample_time, self.x_data)

        # Calculate the error between the original signal and the reconstructed signal
        error = y_data - recontructed_data
        
        # plotting the original signal and the sampled data as dots 
        self.canvas3.axes.plot(self.x_data, y_data)
        self.canvas3.axes.scatter(sample_time, sample_data, color='k', s=10)
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
    
    def sinc_interp(self, sample_data,sample_time , original_time):

        #It's important that the signal values and the corresponding time values have matching lengths for interpolation to be meaningful 
        if len(sample_data) != len(sample_time):
            raise ValueError('sample_data and sample_time must be the same length')

        # Find the period that represents the time or distance between two consecutive samples.

        T = sample_time[1] - sample_time[0]
        # converting to 2D array In signal processing and interpolation, 
        # working with 2D arrays (matrices) often allows for more efficient and vectorized computations
        sincM = np.tile(original_time, (len(sample_time), 1)) - \
            np.tile(sample_time[:, np.newaxis], (1, len(original_time)))
        #calculates a weighted sum of the resampled data x using the sinc function values
        interpolated_data = np.dot(sample_data, np.sinc(sincM/T))
        return interpolated_data
    


    def plotHSlide(self):
        self.canvas3.axes.clear()
        self.canvas4.axes.clear()
        self.canvas5.axes.clear()
        self.plot(self.y_data)

    def update_slider_labels(self):
        selected_option = self.sample_rate_comboBox.currentIndex()
        min_value = self.freq_slider.minimum()
        if selected_option == self.normFreq_index:
            max_value = int(ceil(4 * self.maxFreq))
        else:
            max_value = 60

        self.min_slider_label.setText(f"Min: {min_value}")
        self.max_slider_label.setText(f"Max: {max_value}")

        # Update the slider's range
        self.freq_slider.setMinimum(min_value)
        self.freq_slider.setMaximum(max_value)


    # def toggle_noise(self):
    #     self.noise_flag = not self.noise_flag  
    #     self.add_gaussian_noise(self.y_data)

            
    # def add_gaussian_noise(self,y_data):
    #         #if the noise checkbox is true -> add noise to the signal
    #         if self.noise_flag:
    #             # Calculate the power of the signal -> computes the mean of the squared values of y_data.
    #             signal_power = np.mean(np.square(y_data))

    #             # Calculate the desired noise power based on SNR
    #             noise_power = signal_power / (10**(self.SNR_LVL / 10))

    #             # Generate white Gaussian noise
    #             noise = np.random.normal(0, np.sqrt(noise_power), len(y_data))

    #             # Add noise to the signal
    #             y_noisy = y_data + noise

    #             #plotting the new noisy signal 
    #             print (len(y_noisy), len(y_data))
    #             self.plot (y_noisy)

    #         #else draw the original data given (without noise)
    #         else: self.plot(y_data)


    # def SNR_value_change(self,):
    #         self.SNR_LVL = self.SNR_slider.value()
    #         self.add_gaussian_noise(self,self.y_data)





            
        


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()