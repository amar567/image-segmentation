imlist={'lena.jpg','baboon.tif','peppers.bmp','house.png','jet.tiff','moon.tiff','ariel.tiff','45.gif','27.gif'};


curr_dir=pwd;
temp1=findstr('\TheCode',curr_dir);
read_path=[curr_dir(1:temp1) 'testimages\'];
for i=1:length(imlist)
    imname=imlist{i}
    a=imread([read_path imname]);
    if length(size(a))==3
        a=rgb2gray(a);
    end
    
    temp=findstr('.',imname);
    path=[curr_dir(1:temp1) 'test_results\'];
    imname=[imname(1:temp-1), '_gray.bmp'];
    imwrite(a,[path imname],'bmp');

    
    [freq intensity]=imhist(a);
     
    for lev=2:2:12 
        first=1; last=256;a1=double(a);
        ths=zeros(1,lev-1);
        ths_val=zeros(1,lev);      
        th_in=1;
        t1=cputime;
        
        for ct=1:lev/2-1 % executes for lev > 2
            [high,low,first,last,a1]=thresh_var(first,last,freq,intensity,a1);%,k1,k2);
            ths(th_in)=first;
            ths(th_in+1)=last;
            ths_val(th_in)=high;
            ths_val(th_in+1)=low;
            th_in=th_in+2;
        end

        [high,low,me,a1]=thresh_mean(first,last,freq,intensity,a1);
        ths(th_in)=me;
        ths_val(th_in)=high;
        ths_val(th_in+1)=low;
        a2=uint8(a1);
        t2=cputime;
        time=t2-t1;
        
        ps=cal_psnr(double(a),a1);
        
        lev_str=num2str(lev);
        imname=[imname(1:temp-1), '_',lev_str,'level.bmp'];
        imwrite(a2,[path imname],'bmp');

        
        
        fid=fopen([path 'results.txt'],'a');
        fprintf(fid,'\n\n%s %s %s %d %s\n%s %s %s %d %s','Time for',imname,...
        ':', time ,'seconds.','PSNR for',imname,':',ps,'dB.');
        fprintf(fid,'\n\n%s %s %s','Thresholds for',imname,':');
        fprintf(fid,'%d ',sort(ths_val));
        fprintf(fid,'\n\n%s %s %s','Sub-ranges for',imname,':');
        fprintf(fid,'%d ',sort(ths));
        fclose(fid);
        
        f1_id=fopen([path 'result_psnr.txt'],'a');
        fprintf(f1_id,'\n %s\t%d',imname,ps);
        fclose(f1_id);
        
        f2_id=fopen([path 'result_cputime.txt'],'a');
        fprintf(f2_id,'\n %s\t%d',imname,time);
        fclose(f2_id);
    end
end
