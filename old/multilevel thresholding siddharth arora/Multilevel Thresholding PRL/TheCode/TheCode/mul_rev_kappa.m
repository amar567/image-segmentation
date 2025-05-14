% imlist={'lena.jpg','baboon.tif','peppers.bmp','house.png','jet.tiff','moon.tiff','ariel.tiff'};
% 
% for i=1:49
%     imname=[num2str(i) '.gif'];
%     imlist(i+7)={imname};
% end

imlist={'jet.tiff'};

k1=.75;
k2=1.5;
    
read_path='F:\prl\multithresh_reviw\test_images\';
for i=1:length(imlist)
    imname=imlist{i};
    a=imread([read_path imname]);
    if length(size(a))==3
        a=rgb2gray(a);
    end
    
    temp=findstr('.',imname);
    path='F:\prl\multithresh_reviw\exp_kappa\';%'C:\MATLAB6p5\work\multi_thresh\try_results\';
    imname=[imname(1:temp-1), '_gray.bmp'];
    imwrite(a,[path imname],'bmp');

    
    [freq intensity]=imhist(a);
%     hist_save=[intensity freq];
%     save([path imname(1:temp-1), '_hist.txt'],'hist_save','-ASCII');
    for lev=4:2:20
        first=1; last=256;a1=double(a);
        ths=zeros(1,lev-1);
        th_in=1;
        t1=cputime;
        
        for ct=1:lev/2-1
            [first last a1]=thresh_var1(first,last,freq,intensity,a1,k1,k2);
            ths(th_in)=first;ths(th_in+1)=last;
            th_in=th_in+2;
        end

        [me a1]=thresh_mean(first,last,freq,intensity,a1);
        ths(th_in)=me;
        a2=uint8(a1);
        t2=cputime;
        time=t2-t1;
        
        ps=cal_psnr(double(a),a1);
        
        lev_str=num2str(lev);
        imname=[imname(1:temp-1), '_',lev_str,'level_' num2str(k1) '_' num2str(k2) '.bmp'];
        imwrite(a2,[path imname],'bmp');

        fid=fopen([path 'results_kappa_' num2str(k1) '_' num2str(k2) '.txt'],'a');
        fprintf(fid,'\n\n%s %s %s %d %s\n%s %s %s %d %s','Time for',imname,...
        ':', time ,'seconds.','PSNR for',imname,':',ps,'dB.');
        fprintf(fid,'\n\n%s %s %s %d %d %d %d %d %d %d %d %d %d','Thresholds for',imname,':',sort(ths));
        fclose(fid);
        
        f1_id=fopen([path 'result_psnr_' num2str(k1) '_' num2str(k2) '.txt'],'a');
        fprintf(f1_id,'\n %s\t%d',imname,ps);
        fclose(f1_id);
        
        f2_id=fopen([path 'result_cputime_' num2str(k1) '_' num2str(k2) '.txt'],'a');
        fprintf(f2_id,'\n %s\t%d',imname,time);
        fclose(f2_id);
    end
end