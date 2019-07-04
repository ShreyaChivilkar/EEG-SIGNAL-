clc
clear
dataset = load('C:\Users\Saurabh\Desktop\Shreya\graz_data\dataset_BCIcomp1')
j = 1;
for i = 1:140
    data(:,:,i) = dataset.x_train(385:1152,:,i);
    data_t (:,:,i)= dataset.x_test(385:1152,:,i);
end
%  j = 1;
% for i =1:140
%     for  k =1:768
%         op_mat(j,1) =dataset.y_train(i,1);
%         j = j+1;
%     end
% end
for k = 1:140
 for i = 1:3
    x =  data(:,i,k);    
    [c,l] = wavedec(x,5,'db4');
    [cd1, cd2, cd3, cd4,cd5] = detcoef(c,l,[1,2,3,4,5]);
    A = appcoef(c,l,'db4',5);
    psd_cd2 = pwelch(cd2);
    psd_cd3 = pwelch(cd3);
    psd_cd4 = pwelch(cd4);
    psd_cd5 = pwelch(cd5);
    psd_a5 = pwelch(A);
    
    c_vector= [ psd_cd2',psd_cd3',psd_cd4',psd_cd5',psd_a5'];
 
    y(i,:) = c_vector;
    

 end
y_new =[ y(1,:),y(2,:),y(3,:)];
 ip(k,:) = y_new;
end

for k = 1:140
 for i = 1:3
    x_t =  data_t(:,i,k);    
    [c_t,l_t] = wavedec(x,5,'db4');
    [cd1_t, cd2_t, cd3_t, cd4_t,cd5_t] = detcoef(c_t,l_t,[1,2,3,4,5]);
    A_t = appcoef(c_t,l_t,'db4',5);
    psd_cd2_t = pwelch(cd2_t);
    psd_cd3_t = pwelch(cd3_t);
    psd_cd4_t = pwelch(cd4_t);
    psd_cd5_t = pwelch(cd5_t);
    psd_a5_t = pwelch(A_t);
    
    c_vector_t= [ psd_cd2_t',psd_cd3_t',psd_cd4_t',psd_cd5_t',psd_a5_t'];
 
    y_t(i,:) = c_vector_t;
    

 end
y_new_t =[ y_t(1,:),y_t(2,:),y_t(3,:)];
 ip_t(k,:) = y_new_t;
end


  [coeff,score,latent,tsquared,explained,mu]  = pca(ip);
a = coeff(:,1:71);
 
 input = ip*a;
op_mat = dataset.y_train;
 op = op_mat';
% new = [input',op_mat];

% %  k 
  new = [input,op_mat];
  
  %For test data project data
%   load('final_svm.mat');
  input_t = ip_t*a;
  yfit = svm_76.predictFcn(input);
