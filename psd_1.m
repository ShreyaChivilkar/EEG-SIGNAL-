data_asc = load('C:\Users\Saurabh\Desktop\Shreya\BCICIV_1_asc\BCICIV_calib_ds1a_cnt.txt','ascii') ;
data = [data_asc(:,27),data_asc(:,31),data_asc(:,29)];
data_ascm = load('C:\Users\Saurabh\Desktop\Shreya\BCICIV_1_asc\BCICIV_calib_ds1a_mrk.txt') ;
% lpFilt = designfilt('lowpassiir','FilterOrder',6, ...
%          'PassbandFrequency',33,'PassbandRipple',0.2, ...
%          'SampleRate',100);  
% data_filt = filtfilt(bpFilt,data);
%  data_filt = filtfilt(lpFilt,data);

j = 1;
for i = 1:200
    pos = data_ascm(i,1);
    
    ip_mat(j:j+400-1,:) = data(pos:pos+400-1,:);
   
    j=j+400;
end 

j = 1;
for i = 1:200
   for k = 1:400
    op_mat(j,1) =  data_ascm(i,2);
    j = j+1;
   end
end

for i = 1:3
    x =  ip_mat(:,i);    
    [c,l] = wavedec(x,5,'db4');
    [cd1, cd2, cd3, cd4,cd5] = detcoef(c,l,[1,2,3,4,5]);
    A = appcoef(c,l,'db4',5);
    c_vector= [cd1' , cd2' ,cd3'];
    y(i,:) = c_vector;

end
% h = spectrum.welch;
% t = psd(h,y','Fs',100);
% plot(t);

pxx = pwelch(y');
[coeff,score_p,latent,tsquared,explained,mu]  = pca(pxx);
input = ip_mat * coeff;
new = [input,op_mat];
