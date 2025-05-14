function value=wt_mean(a,b,freq,intensity)
sum_1=0;
sum_2=0;
for i=a:b
    sum_1=sum_1+freq(i)*intensity(i);
    sum_2=sum_2+freq(i);
end

value=sum_1/sum_2;
