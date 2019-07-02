% Classification using SVM classifier
 

clc
clear

%Load Calibration data 

data_asc = load('C:\Users\Saurabh\Desktop\Shreya\BCICIV_1_asc\BCICIV_calib_ds1a_cnt.txt','ascii') ;
% data_asc1 = load('C:\Users\Saurabh\Desktop\Shreya\BCICIV_1_asc\BCICIV_calib_ds1b_cnt.txt','ascii') ;
% Convert the data into required microvolt format after selecting required  electrodes

% data = ( [ data_asc(:,6), data_asc(:,11), data_asc(:,14), data_asc(:,26), data_asc(:,27),...
%           data_asc(:,28), data_asc(:,29), data_asc(:,30), data_asc(:,31), data_asc(:,32), ...
%           data_asc(:,43), data_asc(:,47), data_asc(:,50), data_asc(:,54)]);
%       
      
data = [data_asc(:,11), data_asc(:,15), data_asc(:,29), data_asc(:,27), data_asc(:,31),...
          data_asc(:,26), data_asc(:,32), data_asc(:,25), data_asc(:,33), data_asc(:,36), data_asc(:,39)];
% data1 = ( [ data_asc1(:,6), data_asc1(:,11), data_asc1(:,14), data_asc1(:,26), data_asc1(:,27),...
%           data_asc1(:,28), data_asc1(:,29), data_asc1(:,30), data_asc1(:,31), data_asc1(:,32), ...
%           data_asc1(:,43), data_asc1(:,47), data_asc1(:,50), data_asc1(:,54)]);
bpFilt = designfilt('bandpassiir', 'FilterOrder', 6, ...
                    'HalfPowerFrequency1',8, 'HalfPowerFrequency2', ...
                    30, 'SampleRate', 100, 'DesignMethod', 'butter');
data_filt = filtfilt(bpFilt,data);
% data_filt1 = filtfilt(bpFilt,data1);


      
% Normalizing Data
%  
% an = max(abs(data));
% datan = (data)/max(an);


 % Load markers file  
 
data_ascm = load('C:\Users\Saurabh\Desktop\Shreya\BCICIV_1_asc\BCICIV_calib_ds1a_mrk.txt') ;

% data_ascm1 = load('C:\Users\Saurabh\Desktop\Shreya\BCICIV_1_asc\BCICIV_calib_ds1b_mrk.txt') ;
% Select required input according to the markers cue position.

j = 1;
for i = 1:200
    pos = data_ascm(i,1);
    
    ip_mat(j:j+400-1,:) = data_filt(pos:pos+400-1,:);
   
    j=j+400;
end 
% j = 1;
% for i = 1:200
%     pos = data_ascm1(i,1);
%     
%     ip_mat2(j:j+400-1,:) = data_filt1(pos:pos+400-1,:);
%    
%     j=j+400;
% end 

% According to the cue postion create the output matrix

j = 1;
for i = 1:200
   for k = 1:400
    op_mat(j,1) =  data_ascm(i,2);
    j = j+1;
   end
end
% j = 1;
% for i = 1:200
%    for k = 1:400
%     op_mat2(j,1) =  data_ascm1(i,2);
%     j = j+1;
%    end
% end
% 
% ip_mat = [ip_mat1;ip_mat2];
% 
% op_mat = [op_mat1;op_mat2];
  

% % Apply coloumn wise DWT

for i = 1:11
    x =  ip_mat(:,i);    
    [c,l] = wavedec(x,4,'db1');
    [cd1, cd2, cd3, cd4] = detcoef(c,l,[1,2,3,4]);
    A = appcoef(c,l,'db1',4);
    c_vector= [cd1' , cd2' ,cd3', cd4', A'];
    y(i,:) = c_vector;

end

% Apply PCA
[coeff,score_p,latent,tsquared,explained,mu]  = pca(y');
% 
% % % Explained indicates maximum variance obtained in 1st 10 coloumns.
% % % Extracting data of the required columns only.
% % % 
coeff_new = coeff(:,1:10);

% Projecting the data on the pca coefficients.

x_pca = ip_mat * coeff_new;

X_train = [x_pca(1:64000,:)];
X_test = [x_pca(64001:80000,:)];
Y_train = [op_mat(1:64000,1)];
Y_test = [op_mat(64001:80000,1)];
% 
% % subplot(4,2,1)
% % plot(data);
% % subplot(4,2,2)
% % plot(data_filt);
% % subplot(4,2,3)
% % plot(datan)
% % subplot(4,2,4)
% % plot(ip_mat) 
% % subplot(4,2,5)
% % plot(X_train,Y_train);
% % subplot(4,2,6)
% % plot(X_test,Y_test);
%   
% 

Mdl7 = fitcsvm(X_train,Y_train,'KernelFunction','gaussian','KernelScale','auto',...
                'ClassNames',{'-1','1'},'Standardize',true);
save('Mdl7.mat');
CVMdl7 = crossval(Mdl7,'KFold',3);
c_out = kfoldPredict(CVMdl7);

save('CVMdl7.mat');

sv = Mdl7.SupportVectors;
gscatter(X_train,Y_train)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('A','B','Support Vector')
hold off


