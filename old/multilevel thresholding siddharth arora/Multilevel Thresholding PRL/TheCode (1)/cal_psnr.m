function psnr=cal_psnr(gray,segm) % only double
difference=0;
Summed_error=0;
[m n]=size(gray);
for i=1:m
    for j=1:n
        difference=gray(i,j)-segm(i,j);
        Summed_error=Summed_error+difference*difference;
    end
end
number_pixels=m*n;
Mean_square_error=Summed_error/(number_pixels);
rmse=sqrt(Mean_square_error);
psnr=20*log10(255/rmse);