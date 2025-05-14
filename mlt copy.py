# import packages from env
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
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
    b = int(b)

    b+=1 # python index thing
    sum1 = np.sum(freq[a:b]*intensity[a:b])
    sum2 = np.sum(freq[a:b]*intensity[a:b]*intensity[a:b])
    sum3 = np.sum(freq[a:b])
    
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


def assign_val(mat, hh, hl, ll, lh, high_assign, low_assign):
    hl = int(hl)
    lh = int(lh)
    mat2 = mat.copy()  # Avoid modifying original matrix
    
    
    # # print(hh, hl, ll, lh, high_assign, low_assign)
    
    # Apply high assignment: hl ≤ mat ≤ hh
    high_mask = (mat2 >= hl) & (mat2 <= hh)
    mat2[high_mask] = high_assign

    # Apply low assignment: ll ≤ mat2 ≤ lh
    low_mask = (mat2 >= ll) & (mat2 <= lh)
    mat2[low_mask] = low_assign
    
    mat2 = mat2.astype(np.int16)
    
    return mat2


def thresh_mean(first, last, freq, intensity, im):
    mean3 = wt_mean(first, last, freq, intensity)
    
    mean4 = wt_mean(mean3 + 1, last, freq, intensity)
    mean5 = wt_mean(first, mean3, freq, intensity)

    im = assign_val(im, hh=last, hl=mean3 + 1, ll=first, lh=mean3,high_assign=mean4, low_assign=mean5)

    return mean4, mean5, mean3, im

def thresh_mode(first, last, freq, intensity, im):
    mean3 = mode(first, last, freq, intensity)
    
    mean4 = mode(mean3 + 1, last, freq, intensity)
    mean5 = mode(first, mean3, freq, intensity)

    im = assign_val(im, hh=last, hl=mean3 + 1, ll=first, lh=mean3,high_assign=mean4, low_assign=mean5)

    return mean4, mean5, mean3, im

def thresh_var(first,last,freq,intensity,im):
    mean1=wt_mean(first,last,freq,intensity)
    print(mean1)
    sum_par=cal_variance(first,last,freq,intensity)
    std=np.sqrt(sum_par)
    high=np.floor(mean1+std) 
    low=np.floor(mean1-std)
    
    if high>last:
        high=last
    if low<first:
        low=first
        
    high_assign=wt_mean(high,last,freq,intensity)
    low_assign=wt_mean(first,low,freq,intensity)
    
    im=assign_val(im,last,high,first,low,high_assign,low_assign)
    
    first=low+1
    last=high-1
    
    high_assign = int(high_assign)
    low_assign = int(low_assign)
    
    return(high_assign,low_assign,first,last,im)

def thresh_var_mode(first,last,freq,intensity,im):
    mean1=mode(first,last,freq,intensity)
    print(mean1)
    sum_par=cal_variance(first,last,freq,intensity)
    std=np.sqrt(sum_par)
    high=np.floor(mean1+std) 
    low=np.floor(mean1-std)
    
    if high>last:
        high=last
    if low<first:
        low=first
        
    high_assign=mode(high,last,freq,intensity)
    low_assign=mode(first,low,freq,intensity)
    
    im=assign_val(im,last,high,first,low,high_assign,low_assign)
    
    first=low+1
    last=high-1
    
    high_assign = int(high_assign)
    low_assign = int(low_assign)
    
    return(high_assign,low_assign,first,last,im)

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

imlist = ["aerial.tiff"]
# imlist = ["aerial.tiff","baboon.tiff","boat.tiff","house.tiff","jet.tiff","lena.tiff","moon.tiff","peppers.tiff"]

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
    for lev in range(2, 12, 2):
        first, last = 0, 255
        a1 = gray
        ths = np.zeros(lev - 1, dtype=int)
        ths_val = np.zeros(lev, dtype=int)
        th_in = 0

        start_time = datetime.now()

        thresholds = []
        for ct in range(lev // 2 - 1):  # execute for lev > 2
            high, low, first, last, a1 = thresh_var(first, last, freq, intensity, a1)
            ths[th_in] = first
            ths[th_in + 1] = last
            ths_val[th_in] = high
            ths_val[th_in + 1] = low
            th_in += 2
            thresholds.append([low,high])
            

        high, low, me, a1 = thresh_mean(first, last, freq, intensity, a1)
        ths[th_in] = me
        ths_val[th_in] = high
        ths_val[th_in + 1] = low

        a2 = a1
        elapsed_time = (datetime.now() - start_time).total_seconds()
        ps = cal_psnr(gray, a1)
        
        plt.imsave(f"./test_results/{imname}_{lev}_mean_img.pdf",a2, cmap='gray')
        # cv2.imwrite(f"./test_results/{imname}_{lev}.bmp", a2)
        
        
        write_path = "./"
        outname = f"{imname}_{lev}level.bmp"
        # Image.fromarray(a2).save(os.path.join(write_path, outname))
        # Log to results.txt
        with open(os.path.join(write_path, 'results_mean.txt'), 'a') as f:
            f.write(f"\n\nTime for {outname}: {elapsed_time:.2f} seconds.\n")
            f.write(f"PSNR for {outname}: {ps:.2f} dB.\n")
            f.write(f"Thresholds for {outname}: {' '.join(map(str, sorted(ths_val)))}\n")
            f.write(f"Sub-ranges for {outname}: {' '.join(map(str, sorted(ths)))}\n")

        with open(os.path.join(write_path, 'result_psnr_mean.txt'), 'a') as f:
            f.write(f"\n{outname}\t{ps:.2f}")
        
        # append to calculated psnr
        psnr_mean[0].append(lev)
        psnr_mean[1].append(ps)

        with open(os.path.join(write_path, 'result_cputime_mean.txt'), 'a') as f:
            f.write(f"\n{outname}\t{elapsed_time:.2f}")
    

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
        a1 = gray
        ths = np.zeros(lev - 1, dtype=int)
        ths_val = np.zeros(lev, dtype=int)
        th_in = 0

        start_time = datetime.now()
        

        for ct in range(lev // 2 - 1):  # execute for lev > 2
            high, low, first, last, a1 = thresh_var_mode(first, last, freq, intensity, a1)
            ths[th_in] = first
            ths[th_in + 1] = last
            ths_val[th_in] = high
            ths_val[th_in + 1] = low
            th_in += 2

        thresholds.append([low,high])
        high, low, me, a1 = thresh_mode(first, last, freq, intensity, a1)
        ths[th_in] = me
        ths_val[th_in] = high
        ths_val[th_in + 1] = low

        a2 = a1
        elapsed_time = (datetime.now() - start_time).total_seconds()
        ps = cal_psnr(gray, a1)
        
        # plt.axis('off')
        plt.imsave(f"./test_results/{imname}_{lev}_mode_img.pdf",a2, cmap='gray')
        # cv2.imwrite(f"./test_results/{imname}_{lev}_mode.bmp", a2)
        
        
        write_path = "./"
        outname = f"{imname}_{lev}level.bmp"
        # Image.fromarray(a2).save(os.path.join(write_path, outname))
        # Log to results.txt
        with open(os.path.join(write_path, 'results_mode.txt'), 'a') as f:
            f.write(f"\n\nTime for {outname}: {elapsed_time:.2f} seconds.\n")
            f.write(f"PSNR for {outname}: {ps:.2f} dB.\n")
            f.write(f"Thresholds for {outname}: {' '.join(map(str, sorted(ths_val)))}\n")
            f.write(f"Sub-ranges for {outname}: {' '.join(map(str, sorted(ths)))}\n")

        with open(os.path.join(write_path, 'result_psnr_mode.txt'), 'a') as f:
            f.write(f"\n{outname}\t{ps:.2f}")
        
        # append to calculated psnr
        psnr_mean[0].append(lev)
        psnr_mean[1].append(ps)

        with open(os.path.join(write_path, 'result_cputime_mode.txt'), 'a') as f:
            f.write(f"\n{outname}\t{elapsed_time:.2f}")
    
    
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