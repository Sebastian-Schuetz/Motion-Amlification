# Motion Amplification

Make tiny movements visible, then take a closer look at their frequency content.

A Python desktop application for phase-based video motion amplification and frequency analysis, developed during my bachelor's thesis in medical engineering. Load a recording, choose a frequency band and amplify movements that are otherwise difficult to see.

## What it can do

- **Two motion amplification algorithms:** a Riesz pyramid implementation and a complex steerable pyramid implementation.
- **Adjustable processing:** set the frequency band, amplification factor, image scale and frame range, with additional pyramid and smoothing settings for each algorithm.
- **Frequency analysis:** select a rectangular region, refine it with a threshold mask and inspect time traces and FFT spectra across pyramid levels, including a combined spectrum.
- **Reusable analysis data:** save intermediate signals to HDF5 and reopen them for further analysis without processing the video again.
- **Desktop interface:** video preview, parameter controls and processing progress, built with CustomTkinter.

Both algorithms use classical signal processing; no trained model is required.

## Getting started

Use Python 3.10 or newer with Tkinter available. From the project directory, install the dependencies and launch the application:

```bash
python -m pip install -r requirements.txt
python main.py
```

1. Load a video and choose **Riesz** or **Steerable**.
2. Enter the recording's sampling rate in Hz, the frequency band and the amplification factor. Adjust the scale and frame range as needed.
3. Enable **Frequency Spectrum** if you also want to analyse the extracted signals, then start processing.
4. In the analysis window, select a region and adjust the threshold to plot its time traces and frequency spectra.

Amplified videos are saved as `.avi` files beside the input video. Optional analysis data is saved as `*_spek.h5`. Output filenames include the processing settings.

For useful results, start with a stationary camera, steady lighting and a scene with clear edges. Keep the upper cutoff below half the sampling rate. The spectra describe the extracted image signals; they are not calibrated displacement measurements.

## Credits and references

The underlying motion amplification methods are the work of **Neal Wadhwa, Michael Rubinstein, Frédo Durand and William T. Freeman**. This project adapts their published methods, reference implementations and pseudocode into Python:

- [**Phase-Based Video Motion Processing** (SIGGRAPH 2013)](https://people.csail.mit.edu/nwadhwa/phase-video/) — the basis of the steerable pyramid implementation; includes the original paper and reference code.
- [**Riesz Pyramids for Fast Phase-Based Video Magnification** (ICCP 2014)](https://people.csail.mit.edu/nwadhwa/riesz-pyramid/) — the basis of the Riesz implementation; includes the paper, supplemental code and pseudocode.

Additional inspiration for efficient luminance processing and downsampling came from **Ali Al-Naji, Sang-Heon Lee and Javaan Chahl**, [*An efficient motion magnification system for real-time applications* (2018)](https://doi.org/10.1007/s00138-018-0916-0).

The Python application, interface and analysis workflow build on these contributions, using OpenCV, NumPy, SciPy, CustomTkinter, h5py, pyFFTW, Matplotlib, Seaborn, pandas and Pillow.
