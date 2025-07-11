import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import label
import matplotlib.pyplot as plt
import os

def main():
    # Parameters
    fc = np.arange(10000, 30001, 5000)  # Cutoff frequencies
    fs = 1000000  # Sampling rate
    test = np.random.randn(fs)
    u = np.arange(0.1, 3.0, 0.05)  # Threshold values 

    # Butterworth filter design
    def butter_lowpass_filter(data, cutoff, fs, order=3):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    # Plot original signal
    plt.figure(figsize=(10, 4))
    plt.plot(test[:1000], label='Original Signal', color='black', alpha=0.5)
    plt.title('Original Signal')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'original_signal.png'))
    plt.close()

    ec_values = []
    R_values = []

    # Plot filtered signals
    plt.figure(figsize=(10, 4))
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    for i in range(len(fc)):
        ftest = butter_lowpass_filter(test, fc[i], fs)
        ftest = ftest - np.mean(ftest)
        ftest = ftest / np.std(ftest)

        plt.plot(ftest[:1000], label=f'Filtered Signal (FWHM= {fc[i] / 1000} kHz)', color=colors[i % len(colors)])

        ec = []
        for threshold in u:
            binary_map = ftest > threshold
            cc, num_objects = label(binary_map)
            ec.append(num_objects)

        ec_values.append(ec)

        R = np.sqrt(np.mean(np.diff(ftest)**2)) / np.sqrt(4 * np.log(2))
        R_values.append(R)

    plt.title('Filtered Signals')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'filtered_signals.png'))
    plt.close()

    print(f"Length of fc: {len(fc)}")
    print(f"Length of ec_values: {len(ec_values)}")

    # Plot normalized EC curve
    plt.figure(figsize=(10, 4))
    for i in range(len(ec_values)):
        plt.plot(u, np.array(ec_values[i]) / fs, color=colors[i % len(colors)], label=f'FWHM= {fc[i] / 1000} kHz')

    plt.ylabel('P(Z>u) (Normalized EC)')
    plt.xlabel('Z')
    plt.legend()
    plt.grid(True)
    plt.title('Normalized Euler Characteristic for Different FWHM')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'normalized_ec.png'))
    plt.close()

    # Plot FWHM fit lines
    plt.figure(figsize=(10, 4))
    for i in range(len(ec_values)):
        plt.plot(u, np.array(ec_values[i]) * R_values[i], 'k--', label=f'FWHM Fit Line (FWHM= {fc[i] / 1000} kHz)')

    plt.ylabel('FWHM Fit Line')
    plt.xlabel('Z')
    plt.legend()
    plt.grid(True)
    plt.title('FWHM Fit Lines for Different FWHM')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fwhm_fit_lines.png'))
    plt.close()

if __name__ == "__main__":
    main()
