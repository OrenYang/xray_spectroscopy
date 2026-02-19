"""
================================================================================
spectrum.py  –  X-ray Spectral Analysis Toolkit
================================================================================

OVERVIEW
--------
This module provides the `Spectrum` class for loading, calibrating, and fitting
experimental x-ray spectra against libraries of simulated spectra, as well as
helper functions for loading calibration data.

CLASSES
-------
Spectrum
    Core class representing a single spectrum (experimental or simulated).

    Loading
    -------
    Spectrum()                  Opens a file dialog to select a spectrum file.
    Spectrum("path/to/file")    Loads directly from a .csv or .ppd file.

    Supported input formats:
      - Calibrated CSV   : columns 'energy', 'wavelength', 'intensity'
      - Raw simulation   : whitespace-delimited .ppd or similar, with a 15-line
                           header containing 'Plasma temperature' and 'Mass density'
      - Raw lineout      : headerless CSV with a single intensity column

    Calibration & Processing
    ------------------------
    .rescale()                  Interactive wavelength/energy calibration via
                                ginput peak clicking and spline fitting.
    .reflectivity_calibration() Corrects for crystal reflectivity curve.
    .filter_transmission()      Corrects for filter transmission.
    .subtract_continuum()       Interactively fits and subtracts a bremsstrahlung
                                continuum (exponential fit); also estimates electron
                                temperature from the continuum slope.

    Fitting
    -------
    .fit(comp, axis, noise_sigma)
        Fits this spectrum against a list of simulated Spectrum objects by
        minimizing MSE. Converts MSE to chi-squared and reports confidence
        regions using the Delta chi2 method (2 free parameters: T and rho):

            68% confidence region:  Delta chi2 < 2.30
            95% confidence region:  Delta chi2 < 6.17

        If noise_sigma is not provided, it is estimated from the best-fit
        residuals. For the most rigorous error bars, supply noise_sigma
        estimated from a signal-free region of your detector.

        Also produces a 2D Delta chi2 heatmap with confidence contours if
        simulation labels are formatted as 'T=<value>_R=<value>'.

    Plotting & Saving
    -----------------
    .plot()                     Plots the spectrum; overlays best-fit curve if
                                a fit has been run.
    .save()                     Saves spectrum and best-fit info to CSV.

FUNCTIONS
---------
upload_folder()                 Loads all .ppd/.csv files from a selected folder
                                as a list of Spectrum objects. Useful for loading
                                a simulation library in one call.
load_reflectivity_calibration() Loads a crystal reflectivity CSV and returns an
                                interpolation function.
load_filter_transmission()      Loads a filter transmission .txt file and returns
                                an interpolation function.

TYPICAL WORKFLOW
----------------
    # 1. Load and calibrate experimental spectrum
    exp = Spectrum()
    exp.rescale()
    exp.reflectivity_calibration(load_reflectivity_calibration())
    exp.subtract_continuum()

    # 2. Load simulation library
    sims = upload_folder()

    # 3. Fit and get confidence regions
    exp.fit(sims, axis='energy')

    # 4. Save result
    exp.save()

DEPENDENCIES
------------
    numpy, matplotlib, pandas, scipy

NOTES
-----
- All intensity arrays are normalized to their absolute maximum before fitting.
- Simulation labels must follow the 'T=<val>_R=<val>' format (auto-set when
  loading .ppd files) for the 2D confidence landscape plot to work.
- The Delta chi2 thresholds assume Gaussian noise and that the grid is dense
  enough to sample the chi2 surface well. If your grid is coarse, the true
  minimum may lie between grid points; consider fitting a 2D paraboloid near
  the minimum for sub-grid-spacing precision.
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from tkinter import filedialog, Tk
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
import re
from matplotlib.lines import Line2D


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
            with open(spectrum, 'r') as f:
                header_lines = [next(f) for _ in range(15)]

            # Extract plasma temperature and mass density
            temp_line = next((line for line in header_lines if 'Plasma temperature' in line), None)
            dens_line = next((line for line in header_lines if 'Mass density' in line), None)
            plasma_temp = temp_line.split('=')[-1].strip() if temp_line else 'UnknownT'
            mass_dens = dens_line.split('=')[-1].strip() if dens_line else 'UnknownD'
            label = f"T={plasma_temp}_R={mass_dens}"

            #load data
            df = pd.read_csv(spectrum, skiprows=15, sep='\s+', names=[0, 1, 2, 3])
            self.energy = df[0]
            self.wavelength = 12398 / self.energy
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
        if "best_fit_label" in df.columns:
            # Only the label; user must supply the comparison set if they want the actual object
            self.best_fit_label = df["best_fit_label"].iloc[0]
        else:
            self.best_fit_label = None
        # Stored best-fit info
        if "best_fit_curve" in df.columns:
            # stored as column → array
            self.best_fit_curve = df["best_fit_curve"].to_numpy()
        else:
            self.best_fit_curve = None

        if "best_fit_score" in df.columns:
            self.best_fit_score = df["best_fit_score"].iloc[0]
        else:
            self.best_fit_score = None

    def rescale(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.intensity, label='Spectrum')

        ax.set_xlabel('Arbitrary Units')
        ax.set_ylabel('Intensity')
        ax.set_title('Click calibration points (press ENTER when done)')
        ax.legend()
        plt.tight_layout()
        plt.show(block=False)

        print("Click as many calibration points as you want, then press ENTER...")
        clicked = plt.ginput(n=-1, timeout=-1)
        time.sleep(0.5)
        plt.close()

        if len(clicked) < 2:
            print("At least 2 points required. Calibration aborted.")
            return

        # Extract nearest index for each clicked x, refine peak
        refined_xs = []
        refined_indices = []

        for cx, cy in clicked:
            # nearest x
            nearest = min(self.x, key=lambda x: abs(x - cx))
            idx = np.where(self.x == nearest)[0][0]

            # --- AUTOMATIC  ±10 index refinement ---
            best_idx = idx
            for j in range(-10, 11):
                test = idx + j
                if 0 <= test < len(self.intensity):
                    if self.intensity[test] > self.intensity[best_idx]:
                        best_idx = test

            refined_indices.append(best_idx)
            refined_xs.append(self.x[best_idx])

        print(f"Refined peaks at x = {refined_xs}")

        # Ask user for calibration type
        while True:
            mode = input("Input values in 'wavelength' (Å) or 'energy' (eV)? ").strip().lower()
            if mode in ['wavelength', 'energy']:
                break
            print("Invalid input.")

        # Input corresponding wavelength/energies
        cal_vals = []
        for px in refined_xs:
            if mode == 'wavelength':
                lam = float(input(f"Enter wavelength for peak at x={px:.2f} (Å): "))
                cal_vals.append(lam)
            else:
                E = float(input(f"Enter energy for peak at x={px:.2f} (eV): "))
                cal_vals.append(12398 / E)   # convert to λ

        cal_vals = np.array(cal_vals)
        refined_xs = np.array(refined_xs)
        sort_idx = np.argsort(refined_xs)
        refined_xs = refined_xs[sort_idx]
        cal_vals = cal_vals[sort_idx]

        # Fit λ(x)
        cal_func = PchipInterpolator(refined_xs, cal_vals)

        # Apply calibration
        self.wavelength = cal_func(self.x)
        self.energy = 12398 / self.wavelength
        self.intensity = self.intensity / np.max(np.abs(self.intensity))

        # Save calibrated lineout
        data = pd.DataFrame({
            'wavelength': self.wavelength,
            'energy': self.energy,
            'intensity': self.intensity
        })

        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, 'scaled_lineouts')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, self.label + '.csv')
        data.to_csv(out_path, index=False)

        # Plot calibrated spectrum
        plt.figure()
        plt.plot(self.energy, self.intensity)
        plt.xlabel('Energy [eV]')
        plt.ylabel('Normalized Intensity')
        plt.title(f'{self.label} (spline fit calibration)')
        plt.tight_layout()
        plt.show()

        print(f"Rescaling complete with spline fit.")

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

    def plot(self, comp=None, axis='energy', show_best_fit=True, title=None, xlabel=None, ylabel=None, legend_names=None, atomic_mass=None):
        x, default_xlabel = self._get_axis_data(axis)
        norm_self = self.intensity / np.max(np.abs(self.intensity))
        atomic_mass = atomic_mass or getattr(self, 'atomic_mass', None)

        # Build format dict from best fit label
        fmt = {}
        if self.best_fit_label is not None:
            t_match = re.search(r'T=([\d.eE+\-]+)', self.best_fit_label)
            r_match = re.search(r'R=([\d.eE+\-]+)', self.best_fit_label)
            if t_match:
                fmt['T'] = f'{float(t_match.group(1)):.4g}'
            if r_match:
                R_val = float(r_match.group(1))
                fmt['rho'] = f'{R_val:.4g}'
                if atomic_mass is not None:
                    fmt['ni'] = f'{mass_density_to_ion_density(R_val, atomic_mass):.3e}'

        def apply_fmt(s):
            try:
                return s.format(**fmt)
            except KeyError:
                return s

        # Build legend name list
        all_labels = [self.label]
        if comp is not None:
            if not isinstance(comp, list):
                comp = [comp]
            all_labels += [c.label for c in comp]
        if show_best_fit and self.best_fit_curve is not None:
            all_labels += [f"Best Fit: {self.best_fit_label}"]

        name_iter = iter([apply_fmt(n) for n in legend_names]) if legend_names is not None else iter(all_labels)

        def next_name(fallback):
            try:
                return next(name_iter)
            except StopIteration:
                return fallback

        plt.plot(x, norm_self, label=next_name(self.label))

        if comp is not None:
            for c in comp:
                if not isinstance(c, self.__class__):
                    raise TypeError(f"Expected instance of {self.__class__.__name__}, got {type(c).__name__} instead.")
                x_c, _ = c._get_axis_data(axis)
                norm_c = c.intensity / np.max(np.abs(c.intensity))
                plt.plot(x_c, norm_c, '--', label=next_name(c.label))

        if show_best_fit and self.best_fit_curve is not None:
            norm_best = self.best_fit_curve / np.max(np.abs(self.best_fit_curve))
            plt.plot(x, norm_best, '--', linewidth=2,
                     label=next_name(f"Best Fit: {self.best_fit_label}"))

        plt.xlabel(xlabel if xlabel is not None else default_xlabel)
        plt.ylabel(ylabel if ylabel is not None else 'Normalized Intensity')
        plt.title(title if title is not None else ('Spectra Comparison' if comp else self.label))
        plt.legend()
        plt.tight_layout()
        plt.show()

    def fit(self, comp, axis='energy', mse_threshold=0.10, atomic_mass=None):
        self.atomic_mass = atomic_mass
        if self.energy is None or self.wavelength is None:
            print("Spectrum in arbitrary units. Scale it first.")
            return

        if not isinstance(comp, list):
            comp = [comp]
        for c in comp:
            if not isinstance(c, self.__class__):
                raise TypeError(f"Expected instance of {self.__class__.__name__}, got {type(c).__name__} instead.")

        x_self, label = self._get_axis_data(axis)
        y_self = self.intensity / np.max(np.abs(self.intensity))

        best_score = np.inf
        best_match = None
        best_interp = None
        scores = []

        all_curves = []

        for c in comp:
            x_c, _ = c._get_axis_data(axis)
            y_c = c.intensity / np.max(np.abs(c.intensity))

            try:
                interp = interp1d(x_c, y_c, bounds_error=False, fill_value=0)
                y_c_interp = interp(x_self)
            except Exception as e:
                print(f"Skipping {c.label} due to interpolation error: {e}")
                continue

            y_c_interp = y_c_interp / np.max(np.abs(y_c_interp))
            mse = np.mean((y_self - y_c_interp) ** 2)
            scores.append((mse, c.label))
            all_curves.append((mse, c.label, y_c_interp))  # <-- store curve

            if mse < best_score:
                best_score = mse
                best_match = c
                best_interp = y_c_interp

        if best_match is None:
            print("No valid comparison spectra found.")
            return

        # Threshold on MSE
        in_range = [(mse, lbl) for mse, lbl in scores if mse <= best_score * (1 + mse_threshold)]

        print(f"\nBest match: {best_match.label} (MSE = {best_score:.5f})")
        # Summarise confidence range
        in_range_T = [float(re.search(r'T=([\d.eE+\-]+)', lbl).group(1)) for _, lbl in in_range if re.search(r'T=([\d.eE+\-]+)', lbl)]
        in_range_R = [float(re.search(r'R=([\d.eE+\-]+)', lbl).group(1)) for _, lbl in in_range if re.search(r'R=([\d.eE+\-]+)', lbl)]

        print(f"\n--- Summary ({mse_threshold*100:.0f}% MSE threshold, {len(in_range)} simulations) ---")
        if in_range_T:
            print(f"  Temperature range: {min(in_range_T):.4g} – {max(in_range_T):.4g} eV")
        if in_range_R:
            print(f"  Density range:     {min(in_range_R):.4g} – {max(in_range_R):.4g} g/cc")
        if in_range_R and atomic_mass is not None:
            ni_min = mass_density_to_ion_density(min(in_range_R), atomic_mass)
            ni_max = mass_density_to_ion_density(max(in_range_R), atomic_mass)
            print(f"  Ion density range: {ni_min:.3e} – {ni_max:.3e} cm^-3")


        # Store results
        self.best_fit_label = best_match.label
        self.best_fit_score = best_score
        self.best_fit_curve = best_interp
        self.confidence_range = in_range

        # Plot MSE landscape
        self._plot_mse_landscape(scores, best_score, mse_threshold)

        in_range_set = {lbl for _, lbl in in_range}

        plt.figure()

        for mse, lbl, y_interp in all_curves:
            if lbl in in_range_set and lbl != best_match.label:
                plt.plot(x_self, y_interp, color='peachpuff', alpha=0.8, linewidth=0.8)

        plt.plot(x_self, y_self, label=f'Experiment: {self.label}')
        plt.plot(x_self, best_interp, '--', label=f'Best Sim: {best_match.label}')
        plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + [
            Line2D([0], [0], color='#FFAD7A', linewidth=0.8,
                   label=f'Within 10% MSE ({len(in_range)-1} fits)')
        ])
        plt.xlabel(label)
        plt.ylabel("Normalized Intensity")
        plt.title("Best MSE Fit")
        plt.show()

        return best_match

    def reflectivity_calibration(self, cal):
        if self.energy is None or self.wavelength is None:
            print("Spectrum in arbitrary units. Scale it first.")
            return

        if cal is None:
            print("No filter provided.")
            return

        interp = cal[2]
        correction = interp(self.energy)

        correction[correction == 0] = np.nan
        self.intensity = self.intensity / correction

        # Save calibrated lineout
        data = pd.DataFrame({
            'wavelength': self.wavelength,
            'energy': self.energy,
            'intensity': self.intensity
        })

        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, 'calibrated_lineouts')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, self.label + '.csv')
        data.to_csv(out_path, index=False)

        print(f"Reflectivity calibration applied to {self.label}.")

    def filter_transmission(self, cal):
        if self.energy is None or self.wavelength is None:
            print("Spectrum must be scaled to energy first.")
            return

        if cal is None:
            print("No filter provided.")
            return

        interp = cal[2]
        T = interp(self.energy)

        T[T <= 0] = 1e-12  # prevent div-by-zero
        self.intensity /= T

        # Save calibrated lineout
        data = pd.DataFrame({
            'wavelength': self.wavelength,
            'energy': self.energy,
            'intensity': self.intensity
        })

        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, 'calibrated_lineouts')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, self.label + '.csv')
        data.to_csv(out_path, index=False)

        print(f"Filter transmission applied to {self.label}.")

    def subtract_continuum(self):

        if self.energy is None or self.wavelength is None:
            print("Spectrum must be scaled to energy first.")
            return

        fig, ax = plt.subplots()
        ax.plot(self.energy, self.intensity, label='Spectrum')

        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('Intensity')
        ax.set_title('Select as many ranges as you want (must click even number of points), press ENTER when done')
        ax.legend()
        plt.tight_layout()
        plt.show(block=False)

        print("Select as many ranges as you want (must click even number of points), then press ENTER...")
        clicked = plt.ginput(n=-1, timeout=-1)
        time.sleep(0.5)
        plt.close()

        if len(clicked)==0 or len(clicked) % 2 == 1:
            print("Must select even number of points. Subrtaction aborted.")
            return


        # Extract nearest index for each clicked x, refine peak
        xs = []
        indices = []

        for cx, cy in clicked:
            # nearest x
            nearest = min(self.energy, key=lambda x: abs(x - cx))
            idx = np.where(self.energy == nearest)[0][0]

            indices.append(idx)
            xs.append(self.energy[idx])

        print(f"Points at x = {xs}")

        # --- Build mask for valid (not-selected) points ---
        indices = sorted(indices)
        mask = np.ones(len(self.energy), dtype=bool)

        # exclude every pair range
        for i in range(0, len(indices), 2):
            lo = indices[i]
            hi = indices[i+1]
            mask[lo:hi+1] = False

        # Extract unmasked data
        x_fit = np.asarray(self.energy[mask])
        y_fit = np.asarray(self.intensity[mask])

        if len(x_fit) < 3:
            print("Not enough points left to fit continuum.")
            return

        # --- Fit exponential A * exp(-B*x) ---
        def expo(x, A, B):
            return A * np.exp(-B * x)

        # initial guess
        A0 = np.max(y_fit)
        B0 = 1 / (x_fit[-1] - x_fit[0] + 1e-9)

        try:
            popt, pcov = curve_fit(expo, x_fit, y_fit, p0=[A0, B0])
            A, B = popt

            # Extract slope uncertainty σ_B safely
            if pcov is not None and np.isfinite(pcov[1, 1]):
                sigma_B = np.sqrt(pcov[1, 1])
            else:
                sigma_B = np.nan

            # ---- Temperature Extraction for E in eV ----
            # T_e = 1/B  with uncertainty:  σ_T = |dT/dB| σ_B = σ_B / B^2

            Te_eV = 1.0 / B

            if np.isfinite(sigma_B):
                sigma_Te = sigma_B / (B**2)
            else:
                sigma_Te = np.nan

            print("Estimated electron temperature from continuum (E in eV):")
            print(f"  T_e = {Te_eV:.4g} ± {sigma_Te:.2g} eV")

            # ---- Optional diagnostic plot: log(I) vs energy ----
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(self.energy, np.log(self.intensity),
                     label="log(Intensity)", linewidth=1.2)

            ax2.plot(self.energy,
                     np.log(expo(self.energy, A, B)),
                     linestyle="--",
                     label=f"Log Fit: ln(I)=ln(A)-B E\nB={B:.3g}")

            ax2.set_xlabel("Energy [eV]")
            ax2.set_ylabel("log(Intensity)")
            ax2.set_title(f"Bremsstrahlung Temperature Fit for {self.label}")
            ax2.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print("Curve fit failed:", e)
            return

        # compute continuum on full energy
        continuum = expo(np.asarray(self.energy), A, B)

        # --- Plot original spectrum with fitted exponential ---
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(self.energy, self.intensity, label="Original Spectrum", linewidth=1.2)
        ax.plot(self.energy, self.intensity-continuum, '--',label="Subtracted Spectrum")

        # continuum predicted on full range
        ax.plot(self.energy, continuum, linestyle="--", label=f"Exp Fit: A exp(-B x)")

        # Highlight ignored regions
        for i in range(0, len(indices), 2):
            lo = indices[i]
            hi = indices[i+1]
            ax.axvspan(self.energy[lo], self.energy[hi], color="red", alpha=0.2)

        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Continuum Fit for {self.label}")

        # --- subtract continuum ---
        self.intensity = self.intensity - continuum


        ax.legend()
        plt.tight_layout()
        plt.show()


        print(f"Continuum subtracted from {self.label}. Fit: I = {A:.3g} exp(-{B:.3g} x)")

    def save(self, folder_name="output", save_plot=True, plot_format='png', axis='energy', title=None, xlabel=None, ylabel=None, legend_names=None, atomic_mass=None):
        # --- Prepare output directory ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        # --- Prepare basic data ---
        data = {
            "wavelength": self.wavelength,
            "energy": self.energy,
            "intensity": self.intensity,
        }
        if self.best_fit_label is not None and self.best_fit_curve is not None:
            data["best_fit_label"] = self.best_fit_label
            data["best_fit_score"] = self.best_fit_score
            data["best_fit_curve"] = self.best_fit_curve

        df = pd.DataFrame(data)
        out_path = os.path.join(out_dir, f"{self.label}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved spectrum to {out_path}")

        # --- Save plot ---
        if save_plot:
            atomic_mass = atomic_mass or getattr(self, 'atomic_mass', None)
            x, default_xlabel = self._get_axis_data(axis)
            norm_self = self.intensity / np.max(np.abs(self.intensity))

            # Build format dict
            fmt = {}
            if self.best_fit_label is not None:
                t_match = re.search(r'T=([\d.eE+\-]+)', self.best_fit_label)
                r_match = re.search(r'R=([\d.eE+\-]+)', self.best_fit_label)
                if t_match:
                    fmt['T'] = f'{float(t_match.group(1)):.4g}'
                if r_match:
                    R_val = float(r_match.group(1))
                    fmt['rho'] = f'{R_val:.4g}'
                    if atomic_mass is not None:
                        fmt['ni'] = f'{mass_density_to_ion_density(R_val, atomic_mass):.1e}'

            def apply_fmt(s):
                try:
                    return s.format(**fmt)
                except KeyError:
                    return s

            all_labels = [self.label]
            if self.best_fit_curve is not None:
                all_labels += [f"Best Fit: {self.best_fit_label}"]

            name_iter = iter([apply_fmt(n) for n in legend_names]) if legend_names is not None else iter(all_labels)

            def next_name(fallback):
                try:
                    return next(name_iter)
                except StopIteration:
                    return fallback

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, norm_self, color='black',linewidth=1.2,label=next_name(self.label))

            if self.best_fit_curve is not None:
                norm_best = self.best_fit_curve / np.max(np.abs(self.best_fit_curve))
                ax.plot(x, norm_best, '--', color='#E8402A', linewidth=1.2,
                        label=next_name(f"Best Fit: {self.best_fit_label}"))

            ax.set_xlabel(xlabel if xlabel is not None else default_xlabel)
            ax.set_ylabel(ylabel if ylabel is not None else 'Normalized Intensity')
            ax.set_title(title if title is not None else self.label)
            ax.legend()
            plt.tight_layout()

            plot_path = os.path.join(out_dir, f"{self.label}.{plot_format}")
            fig.savefig(plot_path, format=plot_format, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot to {plot_path}")

    def _plot_mse_landscape(self, scores, best_score, mse_threshold):
        Tvals, Rvals, msevals = [], [], []

        for mse, lbl in scores:
            t_match = re.search(r'T=([\d.eE+\-]+)', lbl)
            r_match = re.search(r'R=([\d.eE+\-]+)', lbl)
            if t_match and r_match:
                Tvals.append(float(t_match.group(1)))
                Rvals.append(float(r_match.group(1)))
                msevals.append(mse)

        if len(Tvals) < 4:
            print("Not enough labeled spectra to plot 2D landscape.")
            return

        Tvals = np.array(Tvals)
        Rvals = np.array(Rvals)
        msevals = np.array(msevals)

        T_unique = np.sort(np.unique(Tvals))
        R_unique = np.sort(np.unique(Rvals))

        grid = np.full((len(R_unique), len(T_unique)), np.nan)
        for t, r, mse in zip(Tvals, Rvals, msevals):
            ti = np.where(T_unique == t)[0][0]
            ri = np.where(R_unique == r)[0][0]
            grid[ri, ti] = mse

        fig, ax = plt.subplots(figsize=(7, 5))
        T_grid, R_grid = np.meshgrid(T_unique, R_unique)
        pcm = ax.pcolormesh(T_grid, R_grid, grid, cmap='viridis_r', shading='auto',
                            norm=LogNorm(vmin=np.nanmin(grid), vmax=np.nanmax(grid)))
        plt.colorbar(pcm, ax=ax, label='MSE')
        ax.set_yscale('log')

        # Contour at threshold
        try:
            cs = ax.contour(T_grid, R_grid, grid, levels=[best_score * (1 + mse_threshold)],
                            colors='white', linestyles='--')
            ax.clabel(cs, fmt=f'{mse_threshold*100:.0f}%% above best')
        except Exception:
            pass

        # Mark best fit
        best_idx = np.argmin(msevals)
        ax.scatter(Tvals[best_idx], Rvals[best_idx], color='orange', zorder=5, s=80, label='Best fit')

        ax.set_xlabel('Temperature [eV]')
        ax.set_ylabel(r'Mass Density [g/cm$^3$]')
        ax.set_title('MSE Landscape')
        ax.legend()
        plt.tight_layout()
        plt.show()


def upload_folder(folder_path=None):
        if folder_path is None:
            root = Tk()
            root.withdraw()
            root.update()
            folder_path = filedialog.askdirectory(title="Select a folder")
            root.destroy()

        objs = []
        for spec in os.listdir(folder_path):
            spec = os.path.join(folder_path, spec)
            if spec.endswith('.ppd') or spec.endswith('.csv'):
                objs.append(Spectrum(spec))
            else:
                continue
        return objs

def load_reflectivity_calibration():
    """
    Opens a file dialog to load a calibration CSV, creates an interpolation
    function, plots the calibration curve, and returns (energies, reflectivity, interp).
    """
    root = Tk()
    root.withdraw()
    root.update()
    calfile = filedialog.askopenfilename(title="Select reflectivity calibration file")
    root.destroy()

    if not calfile:
        print("No calibration file selected.")
        return None, None, None

    # --- Load CSV ---
    cal = pd.read_csv(calfile)
    energies = cal.iloc[:, 0].values
    reflectivity = cal.iloc[:, 1].values

    # --- Create interpolation function ---
    interp = interp1d(
        energies,
        reflectivity,
        bounds_error=False,
        fill_value="extrapolate"
    )

    # --- Plot calibration data ---
    plt.figure(figsize=(7, 5))

    # Original calibration points
    plt.scatter(energies, reflectivity, label="Calibration Points", s=40)

    # Smooth interpolation for plotting
    dense_E = np.linspace(min(energies), max(energies), 400)
    dense_R = interp(dense_E)

    plt.plot(dense_E, dense_R, '--', label="Interpolated Curve")

    plt.xlabel("Energy [eV]")
    plt.ylabel("Reflectivity [mRad]")
    plt.title("Crystal Reflectivity Calibration")
    plt.legend()

    plt.show(block=True)

    print("Calibration file loaded. Interpolation ready.")

    return energies, reflectivity, interp

def load_filter_transmission():
    """
    Loads a filter transmission .txt file in the format:

        C10H8O4 Density=1.4 Thickness=5. microns
        Photon energy (eV), Transmission
        800.00   2.46E-02
        ...

    Returns:
        (energies, transmission, interp)
    """

    # --- File dialog ---
    root = Tk()
    root.withdraw()
    root.update()
    fltfile = filedialog.askopenfilename(
        title="Select filter transmission file",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()

    if not fltfile:
        print("No file selected.")
        return None, None, None

    # --- Read header line 0 (material info) ---
    with open(fltfile, "r") as f:
        header = f.readline().strip()

    # --- Use pandas to read the numeric table ---
    df = pd.read_csv(
        fltfile,
        skiprows=2,           # skip the first two lines (header + column label)
        delim_whitespace=True,
        names=["energy", "transmission"]
    )

    # Convert to numpy
    energies = df["energy"].values
    transmission = df["transmission"].values

    # --- Interpolator ---
    interp = interp1d(
        energies,
        transmission,
        bounds_error=False,
        fill_value="extrapolate"
    )

    # --- Plot for visual confirmation ---
    plt.figure(figsize=(7, 5))
    plt.scatter(energies, transmission, label="Filter data", s=20)

    dense_E = np.linspace(energies.min(), energies.max(), 500)
    dense_T = interp(dense_E)

    plt.plot(dense_E, dense_T, '--', label="Interpolated", linewidth=1.2)

    plt.xlabel("Photon Energy [eV]")
    plt.ylabel("Transmission")
    plt.title(f"Filter Transmission Curve\n{header}")
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

    print(f"Loaded filter: {header}")
    print("Filter ready.")

    return energies, transmission, interp

def mass_density_to_ion_density(rho_gcc, atomic_mass):
    N_A = 6.02214076e23
    return (rho_gcc * N_A) / atomic_mass


if __name__ == "__main__":
    exp = Spectrum('/Users/orenyang/Desktop/for_kim/output/7247.csv')
    sims = upload_folder('/Users/orenyang/Documents/GitHub/xray_spectroscopy/prismspect/ne/ne_50-350eV_1e18-5e21_4mm/results')
    exp.fit(sims, atomic_mass=20.18)
    exp.plot(legend_names=['Measured Spectrum', 'Best Fit: T={T} eV, $n_i$={ni} cm$^{{-3}}$'])
    exp.save(folder_name='test', plot_format='svg', atomic_mass=20.18,
             legend_names=['Measured Spectrum', 'Best Fit: T={T} eV, $n_i$={ni} cm$^{{-3}}$'])
