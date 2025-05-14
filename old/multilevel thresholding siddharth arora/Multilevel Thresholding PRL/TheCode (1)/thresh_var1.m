% MODIFIED ON 23 APRIL 2007. KAPPA INCLUDED.

function [first,last,im]=thresh_var(first,last,freq,intensity,im,k1,k2)
kappa=1;
mean1=wt_mean(first,last,freq,intensity);
sum_par=cal_variance(first,last,freq,intensity);
std=sqrt(sum_par);
high=double(uint8(mean1+k1*std)); 
low=double(uint8(mean1-k2*std)); 
high_assign=wt_mean(high+2,last,freq,intensity);
low_assign=wt_mean(first,low+1,freq,intensity);
im=assign_val(im,last,high+2,first,low+1,high_assign,low_assign);
first=low+2;
last=high+1;