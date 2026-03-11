import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import numpy as np
from tifffile import imread

def main():
    im_dir = '/Users/orenyang/Documents/UCSD_Lab/Cornell/IPs'
    #Loop through a directory with cropped images to create the lineouts
    for f in os.listdir(im_dir):
        img = os.path.join(im_dir,f)
        lineout(img, '/Users/orenyang/Documents/UCSD_Lab/Cornell/apsara_lineout_plot', '/Users/orenyang/Documents/UCSD_Lab/Cornell/apsara_lineout')
    return

#Function to plot and save lineouts
######## CROP AND ROTATE IMAGE TO DESIRED ORIENTATION MANUALLY ########
#Input: img - a single image file of any type, file name must start with 4 digit shot number
def lineout(img, plot_dir='lineout_plots', lineout_dir='lineouts'):
    name = img.split('/')[-1]
    #make output folders
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(lineout_dir, exist_ok=True)
    #Determine image file type to read image and create numpy array
    try:
        if img.endswith('.tif'):
            im = imread(img)
        else:
            im = Image.open(img).convert('L')
    except (ValueError, IndexError, OSError) as e:
        print(f"Skipping {img}: {e}")
        return None
    im_array = np.array(im)
    #Sum columns of image intensity to create lineout
    profile = im_array.sum(axis=0)
    x = np.linspace(0, 1, len(profile))
    #Ensure all lineouts are oriented in the same direction
    if np.mean(profile) > max(profile) / 1.5:
        profile = -1 * profile.astype(np.float64)
    #Plot lineout and image together
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
    #Save lineout as a csv to desired folder (lineout_dir at the top)
    np.savetxt(lineout_dir + '/{}.csv'.format(name[:-4]), profile, delimiter=",")
    return profile

if __name__ == "__main__":
    main()
