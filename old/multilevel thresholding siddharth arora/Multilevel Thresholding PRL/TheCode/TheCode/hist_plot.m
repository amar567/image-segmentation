imlist={'lena.jpg','baboon.tif','peppers.bmp','house.png','jet.tiff','moon.tiff','ariel.tiff'};

for i=1:49
    imname=[num2str(i) '.gif'];
    imlist(i+7)={imname};
end


read_path='I:\prl\multithresh_reviw\test_images\';
for i=1:length(imlist)
    imname=imlist{i};
    a=imread([read_path imname]);
    if length(size(a))==3
        a=rgb2gray(a);
    end
    
    temp=findstr('.',imname);
    path='I:\prl\multithresh_reviw\image_histograms\';
    imname=[imname(1:temp-1), '_hist.jpg'];
%     imwrite(a,[path imname],'bmp');

    
    [freq intensity]=imhist(a);
    hist_save=[intensity freq];
%    save([path imname(1:temp-1), '_hist.txt'],'hist_save','-ASCII');
%    plot(intensity,freq)
    name=[path imname];
    h=plot(intensity,freq,'LineWidth',2);   
    saveas(h,name);
    close;
    
%     sk=sum(( (freq - mean(1,256,freq,intensity)*ones(size(freq))) ./sum(freq) ).^3)/256;
%     fid=fopen([path 'result_skew.txt'],'a');
%     fprintf(fid,'\n %s\t%d',imname,sk);
%     fclose(fid);
end