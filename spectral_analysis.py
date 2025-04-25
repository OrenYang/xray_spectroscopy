import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from tkinter import filedialog, Tk


class Spectrum:
    def __init__(self, spectrum=None, label=None):
        if spectrum is None:
            root = Tk()
            root.withdraw()
            root.update()
            spectrum = filedialog.askopenfilename(title="Select a spectrum file")
            root.destroy()
        if not spectrum:
            raise ValueError("No spectrum file provided or selected.")

        df = pd.read_csv(spectrum)

        if 'energy' in df.columns:
            self.energy = df['energy']
            self.wavelength = df['wavelength']
            self.intensity = df['intensity']
        elif '#' in df.columns[0]:
            df = pd.read_csv(spectrum, skiprows=15,sep='\s+',names=[0,1,2,3])
            self.energy = df[0]
            self.wavelength = 12398/self.energy
            self.intensity = df[1]
        else:
            df = pd.read_csv(spectrum, header=None, skiprows=1)
            self.energy = None
            self.wavelength = None
            self.intensity = np.array(df[0])

        self.x = np.array(df.index.values)
        if label is None:
            label = spectrum.split('/')[-1][:4]
        self.label = label


    def rescale(self):
        name = self.label

        fig, ax = plt.subplots()
        ax.plot(self.x, self.intensity, label='Spectrum')

        ax.set_xlabel('Arbitrary Units')
        ax.set_ylabel('Intensity')
        ax.set_title('Click two peaks to assign wavelengths')
        ax.legend()
        plt.tight_layout()
        plt.show(block=False)

        # Wait for user to click on 2 peaks
        print("Click on two peaks you want to use for calibration...")
        clicked = plt.ginput(2, timeout=-1, show_clicks=True)
        time.sleep(0.5)
        plt.close()

        # Get nearest peak x values to clicked points
        click_xs = [c[0] for c in clicked]
        peak1 = min(self.x, key=lambda x: abs(x - click_xs[0]))
        peak2 = min(self.x, key=lambda x: abs(x - click_xs[1]))

        ind1 = np.where(self.x == peak1)[0][0]
        ind2 = np.where(self.x == peak2)[0][0]

        for j in range(-5, 5):
            test_index = ind1 + j
            if 0 <= test_index < len(self.intensity):
                if self.intensity[test_index] > self.intensity[ind1]:
                    ind1 = test_index
                    peak1 = self.x[ind1]

        for j in range(-5, 5):
            test_index = ind2 + j
            if 0 <= test_index < len(self.intensity):
                if self.intensity[test_index] > self.intensity[ind2]:
                    ind2 = test_index
                    peak2 = self.x[ind2]

        print(f"Selected peaks at: {peak1:.2f} and {peak2:.2f}")
        lambda1 = float(input(f"Enter wavelength for peak at {peak1:.2f}: "))
        lambda2 = float(input(f"Enter wavelength for peak at {peak2:.2f}: "))

        self.wavelength = lambda1 + (self.x - peak1) / (peak2 - peak1) * (lambda2 - lambda1)
        self.intensity = (self.intensity - np.min(self.intensity)) / np.ptp(self.intensity)
        self.energy = 12398 / self.wavelength

        data = pd.DataFrame({
            'wavelength': self.wavelength,
            'energy': self.energy,
            'intensity': self.intensity
        })

        #Save data to scaled_lineouts folder in directory with code
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, 'scaled_lineouts')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, self.label + '.csv')
        data.to_csv(output_path, sep=',', index=False)

        #Plot the scaled and normalized lineout vs energy
        fig, ax = plt.subplots()
        ax.plot(self.energy,self.intensity)
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title('{}'.format(self.label))
        plt.tight_layout()
        plt.show()

    def _get_axis_data(self, axis):
        if not isinstance(axis, str):
            raise ValueError("axis must be a string")

        axis = axis.strip().lower()

        if axis == 'energy':
            x = self.energy
            label = 'Energy [eV]'
        elif axis == 'wavelength':
            x = self.wavelength
            label = 'Wavelength [A]'
        else:
            raise ValueError("axis must be either 'energy' or 'wavelength'")

        if x is None:
            print("Data not scaled. Defaulting to arb. units")
            x = self.x
            label = ''

        return x, label

    def plot(self, axis='energy'):
        x, label = self._get_axis_data(axis)

        plt.plot(x, self.intensity)
        plt.xlabel(label)
        plt.ylabel('Intensity (arb. units)')
        plt.show()

    def compare_plots(self, comp, axis='energy'):
        if not isinstance(comp, self.__class__):
            raise TypeError(f"Expected an instance of {self.__class__.__name__}, got {type(comp).__name__} instead.")

        x, label = self._get_axis_data(axis)
        x_comp, _ = comp._get_axis_data(axis)

        norm_self = (self.intensity - np.min(self.intensity)) / np.ptp(self.intensity)
        norm_comp = (comp.intensity - np.min(comp.intensity)) / np.ptp(comp.intensity)

        plt.plot(x, norm_self, label=self.label)
        plt.plot(x_comp, norm_comp, label=comp.label)
        plt.xlabel(label)
        plt.ylabel('Intensity (arb. units)')
        plt.legend()
        plt.show()

def upload_folder(type = 'experimental'):
    root = Tk()
    root.withdraw()
    root.update()
    dir = filedialog.askdirectory(title="Select a folder")
    root.destroy()

    objs = []

    if type == 'experimental':
        for spec in os.listdir(dir):
            spec = os.path.join(dir,spec)
            objs.append(Spectrum(spec))
    elif type == 'sim':
        for spec in os.listdir(dir):
            spec = os.path.join(dir,spec)
            if spec.endswith('.ppd'):
                objs.append(Spectrum(spec))
            else:
                continue

    return objs
