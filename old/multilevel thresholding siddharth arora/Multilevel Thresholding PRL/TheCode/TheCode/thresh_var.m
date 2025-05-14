function [high_assign,low_assign,first,last,im]=thresh_var(first,last,freq,intensity,im)

mean1=wt_mean(first,last,freq,intensity);
sum_par=cal_variance(first,last,freq,intensity);
std=sqrt(sum_par);
high=double(uint8(mean1+std)); 
low=double(uint8(mean1-std)); 

if high>last
    high=last;
end
if low<first
    low=first;
end
high_assign=uint8(wt_mean(high,last,freq,intensity));
low_assign=uint8(wt_mean(first,low,freq,intensity));
im=assign_val(im,last,high,first,low,high_assign,low_assign);
first=low+1;
last=high-1;