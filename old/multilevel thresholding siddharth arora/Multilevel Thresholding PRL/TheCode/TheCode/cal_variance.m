function varian=cal_variance(a,b,freq,intensity)
sum1=0;
sum2=0;
sum3=0;
for i=a:b
    sum1=sum1+freq(i)*intensity(i);
    sum2=sum2+freq(i)*intensity(i)*intensity(i);
    sum3=sum3+freq(i);
end
varian=sum2/sum3-(sum1/sum3)^2;