import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import numpy as np
from tifffile import imread


def main():
    dir = '/Users/orenyang/Library/CloudStorage/GoogleDrive-oryang@ucsd.edu/Shared drives/HEDP CER/other_experiments/2026_Cornell_gaspuff/data_analysis/fssr/cropped/Ar/gel'
    plot_dir = '/Users/orenyang/Library/CloudStorage/GoogleDrive-oryang@ucsd.edu/Shared drives/HEDP CER/other_experiments/2026_Cornell_gaspuff/data_analysis/fssr/lineout_plots/Ar/line/gel'
    lineout_dir = '/Users/orenyang/Library/CloudStorage/GoogleDrive-oryang@ucsd.edu/Shared drives/HEDP CER/other_experiments/2026_Cornell_gaspuff/data_analysis/fssr/lineouts/Ar/line/gel'
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        line_profiles(f, y_positions=[0.2, 0.4, 0.6, 0.8],
              use_fractions=True, plot_dir=plot_dir, lineout_dir=lineout_dir)
        '''lineout(f,
                plot_dir,
                lineout_dir)'''


# Function to plot and save lineouts
######## CROP AND ROTATE IMAGE TO DESIRED ORIENTATION MANUALLY ########
# Input: img - a single image file of any type, file name must start with 4 digit shot number
def lineout(img, plot_dir='lineout_plots', lineout_dir='lineouts'):
    name = img.split('/')[-1]
    # make output folders
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(lineout_dir, exist_ok=True)
    # Determine image file type to read image and create numpy array
    try:
        if img.endswith('.tif'):
            im = imread(img)
            im_array = np.array(im)
        elif img.endswith('.gel'):
            im_array, im = _read_gel(img)
        else:
            im = Image.open(img).convert('L')
            im_array = np.array(im)
    except (ValueError, IndexError, OSError) as e:
        print(f"Skipping {img}: {e}")
        return None
    # Sum columns of image intensity to create lineout
    profile = im_array.sum(axis=0)
    x = np.linspace(0, 1, len(profile))
    # Ensure all lineouts are oriented in the same direction
    if np.mean(profile) > max(profile) / 1.5:
        profile = -1 * profile.astype(np.float64)
    # Plot lineout and image together
    fig, ax = plt.subplots(2)
    ax[0].imshow(im)
    ax[0].set_title(name[:4])
    ax[0].set_xticks([])
    ax[0].set_yticks([])  # Remove y-axis scale from image
    ax[1].plot(x, profile)
    ax[1].set_xlabel('Energy')
    ax[1].set_ylabel('Intensity')
    ax[1].set_xticks([])
    # Match lineout x-limits exactly to image width — eliminates margin compression
    ax[1].set_xlim(0, 1)
    ax[1].margins(x=0)
    plt.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    plt.savefig(plot_dir + '/{}.png'.format(name[:-4]), bbox_inches='tight')
    plt.show()
    # Save lineout as a csv to desired folder (lineout_dir at the top)
    np.savetxt(lineout_dir + '/{}.csv'.format(name[:-4]), profile, delimiter=",")
    return profile


def line_profiles(img, y_positions, plot_dir='lineout_plots', lineout_dir='lineouts',
                  avg_width=1, use_fractions=False):
    """
    Extract horizontal line profiles at specified y-positions from an image.

    Parameters
    ----------
    img : str
        Path to image file (.tif, .gel, or any PIL-readable format).
    y_positions : list of int or float
        Y-positions at which to extract horizontal profiles.
        - If use_fractions=False (default): pixel row indices (int).
        - If use_fractions=True: fractional positions from 0.0 (top) to 1.0 (bottom).
    plot_dir : str
        Directory to save the combined plot (image + profiles).
    lineout_dir : str
        Directory to save each profile as a CSV.
    avg_width : int
        Number of rows to average around each y-position (default=1, i.e. single row).
        Use e.g. avg_width=5 to average 5 rows centered on the y-position,
        which reduces noise.
    use_fractions : bool
        If True, interpret y_positions as fractions of image height (0.0–1.0).
        If False (default), interpret as absolute pixel row indices.

    Returns
    -------
    profiles : dict
        Keys are the actual pixel y-positions used; values are 1-D numpy arrays.

    Example
    -------
    # Extract profiles at rows 100, 250, and 400 (pixel indices)
    line_profiles('Shot_7680.tif', y_positions=[100, 250, 400],
                  plot_dir='plots', lineout_dir='lineouts', avg_width=3)

    # Same but using fractional positions (top quarter, middle, bottom quarter)
    line_profiles('Shot_7680.tif', y_positions=[0.25, 0.5, 0.75],
                  plot_dir='plots', lineout_dir='lineouts',
                  avg_width=3, use_fractions=True)
    """
    name = img.split('/')[-1]
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(lineout_dir, exist_ok=True)

    # --- Load image ---
    try:
        if img.endswith('.tif'):
            im_array = np.array(imread(img))
        elif img.endswith('.gel'):
            im_array, _ = _read_gel(img)
        else:
            im_array = np.array(Image.open(img).convert('L'))
    except (ValueError, IndexError, OSError) as e:
        print(f"Skipping {img}: {e}")
        return None

    n_rows, n_cols = im_array.shape[:2]
    half = avg_width // 2

    # --- Resolve y-positions to pixel indices ---
    pixel_ys = []
    for y in y_positions:
        if use_fractions:
            py = int(round(float(y) * (n_rows - 1)))
        else:
            py = int(y)
        if not (0 <= py < n_rows):
            print(f"Warning: y={y} (pixel row {py}) is out of bounds for image "
                  f"height {n_rows}. Skipping.")
            continue
        pixel_ys.append(py)

    if not pixel_ys:
        print("No valid y-positions provided.")
        return None

    # --- Extract profiles ---
    profiles = {}
    x = np.linspace(0, 1, n_cols)

    for py in pixel_ys:
        row_start = max(0, py - half)
        row_end   = min(n_rows, py + half + 1)
        # If image is RGB/multi-channel, collapse to grayscale first
        slice_ = im_array[row_start:row_end]
        if slice_.ndim == 3:
            slice_ = slice_.mean(axis=2)
        profiles[py] = slice_.mean(axis=0)

    # --- Plot: image on top, all profiles below ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # Top panel: image with y-position indicators
    display_im = im_array if im_array.ndim == 2 else im_array.astype(np.uint8)
    axes[0].imshow(display_im, aspect='equal', cmap='gray' if im_array.ndim == 2 else None)
    axes[0].set_title(name[:4])
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    colors = plt.cm.tab10(np.linspace(0, 1, len(pixel_ys)))
    for py, color in zip(pixel_ys, colors):
        axes[0].axhline(y=py, color=color, linewidth=1.2, linestyle='--',
                        label=f'y={py}')

    # Bottom panel: all profiles
    for py, color in zip(pixel_ys, colors):
        axes[1].plot(x, profiles[py], color=color, label=f'y={py}')

    axes[1].set_xlabel('Energy')
    axes[1].set_ylabel('Intensity')
    axes[1].set_xticks([])
    axes[1].set_xlim(0, 1)
    axes[1].margins(x=0)
    axes[1].legend(fontsize=8, loc='upper right')

    plt.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f'{name[:-4]}_line_profiles.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()

    # --- Save each profile as its own CSV ---
    for py in pixel_ys:
        csv_path = os.path.join(lineout_dir, f'{name[:-4]}_y{py}.csv')
        np.savetxt(csv_path, profiles[py], delimiter=",")
        print(f"Saved: {csv_path}")

    return profiles


def _read_gel(filepath):
    """
    Read a Fuji .gel file and return pixel values converted to linear PSL-proportional units.
    .gel files are 16-bit big-endian, log-encoded: linear = 10^(pixel * log_range / 65535) - 1
    The log range is typically 5.0 (OD 0–5) for most Fuji scanners (BAS, FLA series).
    """
    try:
        raw = imread(filepath)
    except Exception:
        with open(filepath, 'rb') as f:
            data = f.read()
        raw = np.frombuffer(data, dtype='>u2')
        size = int(np.sqrt(len(raw)))
        raw = raw.reshape(size, size)
    LOG_RANGE = 5.0  # Fuji default; check your scanner settings
    linear = 10 ** (raw.astype(np.float64) * LOG_RANGE / 65535) - 1
    return linear, raw


if __name__ == "__main__":
    main()
