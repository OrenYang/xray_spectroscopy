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

    def plot(self, comp=None, axis='energy',show_best_fit=True):
        x, label = self._get_axis_data(axis)
        norm_self = self.intensity / np.max(np.abs(self.intensity))
        plt.plot(x, norm_self, label=self.label)

        if comp is not None:
            if not isinstance(comp, list):
                comp = [comp]
            for c in comp:
                if not isinstance(c, self.__class__):
                    raise TypeError(f"Expected instance of {self.__class__.__name__}, got {type(c).__name__} instead.")
                x_c, _ = c._get_axis_data(axis)
                norm_c = c.intensity / np.max(np.abs(c.intensity))
                plt.plot(x_c, norm_c, '--', label=c.label)

        if show_best_fit and self.best_fit_curve is not None:
                # normalize for consistency
                norm_best = self.best_fit_curve / np.max(np.abs(self.best_fit_curve))
                plt.plot(x, norm_best, '--', linewidth=2,
                         label=f"Best Fit: {self.best_fit_label}")

        plt.xlabel(label)
        plt.ylabel('Normalized Intensity')
        plt.legend()
        plt.title('Spectra Comparison' if comp else self.label)
        plt.show()

    def fit(self, comp, axis='energy'):

        if self.energy is None or self.wavelength is None:
            print("Spectrum in arbitrary units. Scale it first.")
            return

        if not isinstance(comp, list):
            comp = [comp]
        for c in comp:
            if not isinstance(c, self.__class__):
                raise TypeError(f"Expected instance of {self.__class__.__name__}, got {type(c).__name__} instead.")

        # Get axis values and normalize self
        x_self, label = self._get_axis_data(axis)
        y_self = self.intensity / np.max(np.abs(self.intensity))

        best_score = np.inf
        best_match = None
        best_interp = None
        scores = []

        for c in comp:
            x_c, _ = c._get_axis_data(axis)
            y_c = c.intensity / np.max(np.abs(c.intensity))

            # Interpolate sim spectrum to experimental axis
            try:
                interp = interp1d(x_c, y_c, bounds_error=False, fill_value=0)
                y_c_interp = interp(x_self)
            except Exception as e:
                print(f"Skipping {c.label} due to interpolation error: {e}")
                continue

            y_c_interp = y_c_interp / np.max(np.abs(y_c_interp))

            mse = np.mean((y_self - y_c_interp) ** 2)
            scores.append((mse, c.label))

            if mse < best_score:
                best_score = mse
                best_match = c
                best_interp = y_c_interp

        if best_match is None:
            print("No valid comparison spectra found.")
            return

        # ---- Store best-fit info inside this object ----
        self.best_fit_label = best_match.label
        self.best_fit_score = best_score
        self.best_fit_curve = best_interp  # y_c evaluated on x_self

        print(f"Best match: {best_match.label} (MSE = {best_score:.5f})")

        # Plot comparison
        plt.plot(x_self, y_self, label=f'Experiment: {self.label}')
        plt.plot(x_self, best_interp, '--', label=f'Best Sim: {best_match.label}')
        plt.xlabel(label)
        plt.ylabel("Normalized Intensity")
        plt.title("Best MSE Fit")
        plt.legend()
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

    def save(self, folder_name="output"):
        """
        Saves the spectrum (energy, wavelength, intensity) to a CSV file
        in a folder named `folder_name`, and includes the best-fit info
        if available.
        """

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

        # --- Add best-fit info when available ---
        if self.best_fit_label is not None and self.best_fit_curve is not None:
            data["best_fit_label"] = self.best_fit_label
            data["best_fit_score"] = self.best_fit_score
            data["best_fit_curve"] = self.best_fit_curve

        df = pd.DataFrame(data)

        # --- Save as CSV ---
        out_path = os.path.join(out_dir, f"{self.label}.csv")
        df.to_csv(out_path, index=False)

        print(f"Saved spectrum to {out_path}")



def upload_folder():
    root = Tk()
    root.withdraw()
    root.update()
    dir = filedialog.askdirectory(title="Select a folder")
    root.destroy()

    objs = []

    for spec in os.listdir(dir):
        spec = os.path.join(dir,spec)
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
