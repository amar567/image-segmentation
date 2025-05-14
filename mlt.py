# import packages from env
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
os.environ["QT_QPA_PLATFORM"] = "offscreen"


# Function for calculating psnr

def cal_psnr(gray,segm):
    summed_error = 0
    m,n = np.shape(gray)

    diff_mat = gray - segm
    sq_diff_mat = diff_mat * diff_mat
    summed_error = np.sum(sq_diff_mat.flatten())
    
    number_pixels = m*n
    mean_square_error = summed_error/number_pixels
    rmse =  np.sqrt(mean_square_error)
    psnr = 20*np.log10(255/rmse)

    return psnr


def cal_variance(a,b,freq,intensity):
    a = int(a)
    b = int(b)+1

    b+=1 # python index thing
    sum1 = np.sum(freq[a:b]*intensity[a:b])
    sum2 = np.sum(freq[a:b]*intensity[a:b]*intensity[a:b])
    sum3 = np.sum(freq[a:b])
    
    print(sum1,sum2,sum3)
    
    variance = sum2/sum3 - (sum1/sum3)**2
    return variance

def mode(a,b,freq,intensity):
    # trying mode insteda
    a=int(a)
    b=int(b)
    # print(a,b)
    if a<b:
        return intensity[a:b+1][np.argmax(freq[a:b+1])]
    elif(a==b):
        return intensity[a-1]
    else:
        return intensity[b:a+1][np.argmax(freq[b:a+1])]


def wt_mean(a,b,freq,intensity):
    
    # original mean calculation
    a=int(a)
    b=int(b)
    sum1 = np.sum(freq[a:b+1]*intensity[a:b+1])
    sum2 = np.sum(freq[a:b+1])
    value = sum1/sum2 
    return int(value)


def assign_val(mat, hh, hl, ll, lh, higher_thresh_col_assign, lower_thresh_col_assign):
    mat2 = mat.copy()  # Avoid modifying original matrix
    
    # Apply b assignment: hl ≤ mat ≤ hh
    b_mask = (mat2 >= hl) & (mat2 <= hh)
    mat2[b_mask] = higher_thresh_col_assign

    # Apply a assignment: ll ≤ mat2 ≤ lh
    a_mask = (mat2 >= ll) & (mat2 <= lh)
    mat2[a_mask] = lower_thresh_col_assign
    
    mat2 = mat2.astype(np.int16)
    
    return mat2


def thresh_mean(first, last, freq, intensity, im):
    # weighted mean of the given range
    mean_range = wt_mean(first, last, freq, intensity)
    # Weighted mean of the values higher than this mean cuttoff
    mean_higher = wt_mean(mean_range + 1, last, freq, intensity)
    # Weighted mean of the values lower than this mean cuttoff
    mean_lower = wt_mean(first, mean_range, freq, intensity)

    im = assign_val(im, hh=last, hl=mean_range + 1, ll=first, lh=mean_range,higher_thresh_col_assign=mean_higher, lower_thresh_col_assign=mean_lower)

    return mean_range, im

def thresh_mode(first, last, freq, intensity, im):
    range_mode = mode(first, last, freq, intensity)
    
    higher_mode = mode(range_mode + 1, last, freq, intensity)
    lower_mode = mode(first, range_mode, freq, intensity)

    im = assign_val(im, hh=last, hl=range_mode + 1, ll=first, lh=range_mode,higher_thresh_col_assign=higher_mode, lower_thresh_col_assign=lower_mode)

    return range_mode, im

def thresh_var(first,last,freq,intensity,im):
    range_mean=wt_mean(first,last,freq,intensity)
    # print(range_mean)
    sum_par=cal_variance(first,last,freq,intensity)
    std=np.sqrt(sum_par)
    b=np.floor(range_mean+std) 
    a=np.floor(range_mean-std)
    
    # adjust if out of range
    if b>last:
        b=last
    if a<first:
        a=first
        
    higher_thresh_col_assign=wt_mean(b,last,freq,intensity)
    lower_thresh_col_assign=wt_mean(first,a,freq,intensity)
    
    im=assign_val(im,last,b,first,a,higher_thresh_col_assign,lower_thresh_col_assign)
    
    new_first=a+1
    new_last=b-1
    
    higher_thresh_col_assign = int(higher_thresh_col_assign)
    lower_thresh_col_assign = int(lower_thresh_col_assign)
    
    return(higher_thresh_col_assign,lower_thresh_col_assign,new_first,new_last,im)

def thresh_var_mode(first,last,freq,intensity,im):
    range_mode=mode(first,last,freq,intensity)
    # print("hi")
    sum_par=cal_variance(first,last,freq,intensity)
    std=np.sqrt(sum_par)
    b=np.floor(range_mode+std) 
    a=np.floor(range_mode-std)
    
    # adjust if out of range
    if b>last:
        b=last
    if a<first:
        a=first
        
    higher_thresh_col_assign=mode(b,last,freq,intensity)
    lower_thresh_col_assign=mode(first,a,freq,intensity)
    
    im=assign_val(im,last,b,first,a,higher_thresh_col_assign,lower_thresh_col_assign)
    
    new_first=int(a+1)
    new_last=int(b-1)
    
    higher_thresh_col_assign = int(higher_thresh_col_assign)
    lower_thresh_col_assign = int(lower_thresh_col_assign)
    
    return(higher_thresh_col_assign,lower_thresh_col_assign,new_first,new_last,im)

# initialize latex file
with open('./compile pdf/compiled_pdf.tex', 'w') as f:
    f.write("""
\\documentclass{article}
\\usepackage{graphicx}
\\usepackage[paperheight=40in,paperwidth=30in]{geometry}
 \\geometry{
%  a4paper,
 total={170mm,257mm},
 left=3mm,
 right=3mm,
 top=3mm,
 bottom=3mm,
 }
\\usepackage{rotating}
\\usepackage{adjustbox}

\\begin{document}
""")



# Image list

# imlist = ["house.tiff"]
imlist = ["aerial.tiff","baboon.tiff","boat.tiff","house.tiff","jet.tiff","lena.tiff","moon.tiff","peppers.tiff"]

for imname in imlist:
    
    image = cv2.imread(f'./testimages/{imname}')
    imname = imname.split(".")[0]
    print(imname)

    pre_gray = image * np.array([ 0.114020904255103,0.587043074451121,0.298936021293775])
    
    gray = np.sum(pre_gray, -1)
    freq, intensity = np.histogram(gray,range=(0,255),bins=255,density=False)
    
    
    temp_int_for_plot = intensity
    # after plotting adjust the size
    intensity = intensity[:-1]

    # ------------------------------- mean -----------------------------------#
    

    psnr_mean = [[],[]]
    thresholds = []
    for lev in range(2, 12, 2):
        first, last = 0, 255
        image_iter = gray

        a,b = 0,0
        for ct in range(lev // 2):  # execute for lev > 2
            b, a, first, last, image_iter = thresh_var(first, last, freq, intensity, image_iter)
            # print(b, a, first, last)

        # print("new a,b", first, last)
        thresholds.append([first,last])
        # print(thresholds)

        mean, image_iter = thresh_mean(first, last, freq, intensity, image_iter)

        ps = cal_psnr(gray, image_iter)
        
        plt.imsave(f"./test_results/{imname}_{lev}_mean_img.pdf",image_iter, cmap='gray')
        
        # append to calculated psnr
        psnr_mean[0].append(lev)
        psnr_mean[1].append(ps)
    

    plt.figure(imname)
    plt.stairs(freq, temp_int_for_plot)
    # print(thresholds)
    colors = ["red","green","blue","orange","pink"]
    handles = []
    for index,level in enumerate(thresholds):
        # print(index,level)
        for thresh in level:
            color = colors[index]
            plt.axvline(x=thresh,label=f'level {index+1} ', linestyle='-',color=color)
        handles.append(plt.Line2D([0], [0], color=color, lw=2, label=f'Level {2*index + 2}')) 
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.legend(handles=handles)
    plt.title(f"Mean thresholds ({imname})")
    plt.savefig(f"./test_results/{imname}_mean.pdf")
    plt.savefig(f"./test_results/{imname}_mean.png")
    plt.close()

    plt.figure(imname+"psnr")
    plt.title(f"Level vs PSNR ({imname})")
    plt.xlabel("Number of thresholds (n)")
    plt.ylabel("PSNR (dB)")
    plt.plot(psnr_mean[0],psnr_mean[1])
    plt.savefig(f"./test_results/{imname}_mean_psnr.pdf")
    plt.close()
    
    # ------------------------------- mode -----------------------------------#

    thresholds = []
    psnr_mean = [[],[]]
    for lev in range(2, 12, 2):
        first, last = 0, 255
        image_iter = gray

        for ct in range(lev // 2 ):  # execute for lev > 2
            b, a, first, last, image_iter = thresh_var_mode(first, last, freq, intensity, image_iter)

        thresholds.append([first,last])
        mode_range, image_iter = thresh_mode(first, last, freq, intensity, image_iter)

        ps = cal_psnr(gray, image_iter)
        
        plt.imsave(f"./test_results/{imname}_{lev}_mode_img.pdf",image_iter, cmap='gray')
            
        # append to calculated psnr
        psnr_mean[0].append(lev)
        psnr_mean[1].append(ps)
    
    
    plt.figure(f'{imname}_new')
    plt.stairs(freq, temp_int_for_plot)
    # print(thresholds)
    colors = ["red","green","blue","orange","pink"]
    handles = []
    for index,level in enumerate(thresholds):
        # # print(index,level)
        for thresh in level:
            color = colors[index]
            plt.axvline(x=thresh,label=f'level {index+1} ', linestyle='-',color=color)
        handles.append(plt.Line2D([0], [0], color=color, lw=2, label=f'Level {2*index + 2}')) 
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.legend(handles=handles)
    plt.title(f"Mode thresholds ({imname})")
    plt.savefig(f"./test_results/{imname}_mode.pdf")
    plt.savefig(f"./test_results/{imname}_mode.png")
    plt.close()
    
    
    plt.figure(imname+"psnr_mode")
    plt.title(f"Level vs PSNR ({imname})")
    plt.xlabel("Number of thresholds (n)")
    plt.ylabel("PSNR (dB)")
    plt.plot(psnr_mean[0],psnr_mean[1])
    plt.savefig(f"./test_results/{imname}_mode_psnr.pdf")
    plt.close()
    
    
    # ------------- latex -------------------
    
    with open('./compile pdf/compiled_pdf.tex', 'a') as f:
        f.write(f"""
    \\fontsize{{32}}{{32}}\\selectfont % Sets font size to 14pt with 16pt line spacing
    \\vspace*{{12mm}}
    \\centerline{{\\textbf{{{imname}}}}}
    \\vspace*{{12mm}}
    \\normalsize % Resets to the base font size
    \\def\\arraystretch{{0.2}} % Adjust row spacing if needed
    \\fontsize{{12}}{{12}}

    \\begin{{tabular}}{{ @{{}}c@{{\\hspace{{0pt}}}}ccccccc }}
        \\multicolumn{{1}}{{c}}{{}} &
            \\textbf{{Thresholds}} &
            \\textbf{{PSNR-vs-Level}} &
            \\textbf{{Level 2}} &
            \\textbf{{Level 4}} &
            \\textbf{{Level 6}} &
            \\textbf{{Level 8}} &
            \\textbf{{Level 10}} \\\\
        \\vspace*{{10mm}}
        \\adjustbox{{valign=c}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{ {imname} mean }}}}}} &
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_mean.pdf}}}}&
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_mean_psnr.pdf}}}}&
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_2_mean_img.pdf}}}}&
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_4_mean_img.pdf}}}}&
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_6_mean_img.pdf}}}}&
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_8_mean_img.pdf}}}}&
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_10_mean_img.pdf}}}}\\\\
        \\vspace*{{6mm}}
        \\adjustbox{{valign=c}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{ {imname} mode }}}}}} &
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_mode.pdf}}}}&
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_mode_psnr.pdf}}}}&
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_2_mode_img.pdf}}}}&
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_4_mode_img.pdf}}}}&
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_6_mode_img.pdf}}}}&
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_8_mode_img.pdf}}}}&
            \\adjustbox{{valign=c}}{{\\includegraphics[width=0.13\\columnwidth]{{../test_results/{imname}_10_mode_img.pdf}}}}\\\\
    \\end{{tabular}}
    \\bigskip
                
        """)
        
with open('./compile pdf/compiled_pdf.tex', 'a') as f:
    f.write("""\end{document}""")