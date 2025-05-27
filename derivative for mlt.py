import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
os.environ["QT_QPA_PLATFORM"] = "offscreen"


# Image list

# imlist = ["aerial.tiff"]
# imlist = ["baboon.tiff"]
# imlist = ["boat.tiff"]
# imlist = ["house.tiff"]
# imlist = ["jet.tiff"]
imlist = ["lena.tiff"]
# imlist = ["moon.tiff"]
# imlist = ["peppers.tiff"]
# imlist = ["aerial.tiff","baboon.tiff","boat.tiff","house.tiff","jet.tiff","lena.tiff","moon.tiff","peppers.tiff"]

# some constants
min_distSpan = 8
n_ra = 6
n_ra1 = 1
n_ra2 = 1

# ----------- auxilary functions ---------------------
# Calculate the running average
def running_average(data, window_size):
    if len(data) < window_size:
        raise ValueError("Window size cannot be larger than the data size.")
    
    # Use convolution for efficient calculation
    weights = np.repeat(1.0, window_size) / window_size
    unpadded_enveolope = np.convolve(data, weights, 'valid')
    # print(np.zeros(window_size))
    padded_enveolope = np.concatenate([np.linspace(0,unpadded_enveolope[0],window_size//2),unpadded_enveolope,np.linspace(unpadded_enveolope[-1],0,window_size//2)])
    return padded_enveolope

# detecting sign change from -ve to +ve
def detect_signChange(fdr,sdr):
    arr = fdr
    # checking sign change by multiplying subsequent elements
    base_arr = np.concatenate([arr,np.zeros(1)])
    one_shifted_arr = np.concatenate([np.zeros(1),arr])
    
    out_arr = base_arr*one_shifted_arr
    change_arr = (out_arr <=0) # sign change occurs when product is -ve
    
    # intensities at which sign change occur
    change_pos = np.where(change_arr == True)[0]
    info_sdr = np.where(sdr >= 0)[0] # these are places where decond derivative is +ve
    
    # when both these condition are satisfied, we get locations of minimas
    # out_pos = change_pos
    out_pos = np.intersect1d( change_pos , info_sdr )
    # dist_info = np.concatenate([out_pos,np.zeros(1)]) - np.concatenate([np.zeros(1),out_pos])
    # bins = np.arange(0,256,1)
    # # print(bins)
    # print(np.histogram(dist_info,bins=bins)[0])
    # magic_number = np.where(np.histogram(dist_info,bins=bins)[0] == 0)[0][2]
    # # magic_number = np.int16(np.abs(magic_number*(1/1-(np.sum(np.abs(np.diff(fdr)))/255**2))))
    # print(magic_number)
    return(out_pos)

def separate_minimas_from_change(signChangeMat,minimumDistributionSpan):
    # if the sign change has occured too fast that is we are in flat fluctuations regions then remove this region
    dist_mat = np.concatenate([signChangeMat,np.zeros(1)]) - np.concatenate([np.zeros(1),signChangeMat])

    outMat = signChangeMat[np.where(dist_mat >= minimumDistributionSpan)[0]]
    
    outMat = np.concatenate([np.array([0]),outMat,np.array([255])])
    
    # print(outMat)
    return outMat
    
def generate_random_hex_color():
    """Generates a random hex color code."""
    hex_color = '#'
    for _ in range(6):
        hex_color += random.choice('0123456789abcdef')
    return hex_color    


import plotly.graph_objects as pltly
import plotly.io as pio
from plotly.offline import plot
import plotly.express as px

# main thresholding logic

for imname in imlist:

    # read the image
    image = cv2.imread(f'./testimages/{imname}')
    imname = imname.split(".")[0]
    print(imname)

    # this step converts image to gray scale according to https://doi.org/10.1006/gmip.1995.1037
    pre_gray = image * np.array([ 0.114020904255103,0.587043074451121,0.298936021293775])    
    gray = np.sum(pre_gray, -1)
    # gray = image[ :,0]
    print(np.shape(image))

    # get the distribution
    # gray = image
    freq, intensity = np.histogram(gray,range=(0,255),bins=255,density=False)
    
    # we the above step has the boundaries for the freq like 234-235 etc so we need to remove thwe last element for plotting
    temp_int_for_plot = intensity
    # after plotting adjust the size
    intensity = intensity[:-1]

    # smoothening of the distribution
    
    # freq = gaussian_filter1d(freq, n_ra)
    # polyorder=3
    # freq = savgol_filter(freq, n_ra, polyorder)
    freq = running_average(freq, n_ra)
    plt.plot(freq,label=f'Running Average ({n_ra})')
    
    # calculate the first and second differences     
    firstDerivative = np.concatenate([freq,np.zeros(1)]) - np.concatenate([np.zeros(1),freq])
    firstDerivative = running_average(firstDerivative,n_ra1)
    
    secDerivative = np.concatenate([firstDerivative,np.zeros(1)]) - np.concatenate([np.zeros(1),firstDerivative])
    secDerivative = running_average(secDerivative,n_ra2)

        
    thresholds = separate_minimas_from_change(detect_signChange(firstDerivative,secDerivative),min_distSpan)

    # Create a scatter plot with lines connecting points
    scatter = pltly.Scatter3d(
        x=freq,
        y=firstDerivative,
        z=secDerivative,
        mode='lines'  # Use 'lines+markers' to connect points with lines
    )

    # Create layout
    layout = pltly.Layout(
        scene=dict(
            aspectmode='cube'  # Keep axes proportions equal
        ),
        height=500,
        # title=f'r vs dr vs d2r for ({direction}_{str(start_time)}-{str(end_time)}_{coord_npy_file_path})'
    )

    # Create the figure
    fig = pltly.Figure(data=[scatter], layout=layout)

    # Show the plot
    # fig.show()
    # plot(fig)
    pio.write_html(fig,"lena_x_dx_d2x.html")
    # fig.write_image(filename+".png", width=1920, height=1080)