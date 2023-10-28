# SBME_DSP_SignalReconstruction
## Task info
Course: Digital Signal Processing
Ph.D Tamer Basha
for third year (first semester)
Date: 28/10/2023
- **Contributors**:
- Camellia Marwan
- Hager Samir
- Farah Ossama
- Malak Nasser

## **Introduction** :-
This PyQt-powered application serves as a **sampling studio**. A desktop application that illustrates the signal sampling and recovery, showing the importance and validation of the Nyquist rate, as well as adding Gaussian noise so you can see its immediate effect on the recovered signal


## Features overview :-
- **Adding Sinusoidal components of different frequencies:** Generate a summed signal from 2 or more signals by selecting their amplitudes, frequencies, and phase shifts. Then, by pressing the 'Show Signal' button, it displays the one you synthesized and adds it to the summed signals graph.
   ![image 1](https://github.com/malaknasser812/SBME_DSP_SignalReconstruction/assets/115308809/d62e95ff-d6e8-471a-aff8-dae05312ba0c)

- **Import signals as csv file:** You can load a custom signal from your computer and subsequently display it to apply the sampling technique.
 ![image](https://github.com/malaknasser812/SBME_DSP_SignalReconstruction/assets/115308809/6ecad6ff-c384-42c2-b189-b504ecc98f48)

- **Sample & Recover:** The user has the capability to manipulate both the sampling and signal reconstruction processes by selecting an appropriate sampling frequency. This choice can be either a normalized frequency up to the signal's f_max or a specific, user-defined value
 ![image](https://github.com/malaknasser812/SBME_DSP_SignalReconstruction/assets/115308809/78139fc8-95cd-45f2-8293-c06bc19176a4)

- **Reconstructing the signal using sinc interpolation:**
 ![image](https://github.com/malaknasser812/SBME_DSP_SignalReconstruction/assets/115308809/23b3ecb4-5ba2-40ce-a4b6-be10c7a88964)


- **Adding Noise:** The application allows the user to add white gaussian noise with a controllable SNR, and then toggle it if they want to deal with the original signal
 ![image](https://github.com/malaknasser812/SBME_DSP_SignalReconstruction/assets/115308809/853e2c17-f74f-4cc9-b28c-85a0602d5b9a)

## Run The Project
You need to install Python 3 and any python IDE on your computer.
- [Download Python 3](https://www.python.org/downloads/)
- Download VS code
1. Clone the repository
```sh
   git clone https://github.com/malaknasser812/SBME_DSP_SignalReconstruction.git
 ```
2. Install project dependencies
```sh
   pip install typing
   pip install os
   pip install PyQt5
   pip install pandas
   pip install numpy
   pip install pyqtgraph
   pip install matplotlip
   pip install scipy
 ```
3. Run the application
```sh
   python main.py
```
## Libraries
- PyQt5
- pyqtgraph
- numpy
- pandas

