function [mean4,mean5,mean3,im]=thresh_mean(first,last,freq,intensity,im)
mean3=wt_mean(first,last,freq,intensity);
mean3=double(uint8(mean3));
mean4=uint8(wt_mean(mean3+1,last,freq,intensity));
mean5=uint8(wt_mean(first,mean3,freq,intensity));
im=assign_val(im,last,mean3+1,first,mean3,mean4,mean5);
