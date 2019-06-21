clc
clear

data_asc = load('C:\Users\DELL\Documents\MATLAB\BCICIV_calib_ds1a_cnt.txt','-ascii') ;
data=data_asc(1:4096,:);
sz=size(data);
% plot(data);
disp(sz)

% Apply DWT and find coefficients of details and approximations
for i = 1:59
    x =  data(1:4096,i);    
    [c,l] = wavedec(x,4,'db1');
    [cd1, cd2, cd3, cd4] = detcoef(c,l,[1,2,3,4]);
    A = appcoef(c,l,'db1',4);
 
%     disp('l')
%     disp(l)

%     disp('cd1')
%     disp(cd1)
    
%     disp('cd2')
%     disp(cd2)
%     
%     disp('cd3')
%     disp(cd3)
%     
%     disp('cd4')
%     disp(cd4)
%     
%     disp('an')
%     disp(A)
%     
%     disp('c')
%     disp(c)
    
%     disp('end')
  
    cd1t = transpose(cd1);
    cd2t = transpose(cd2);
    cd3t = transpose(cd3);
    cd4t = transpose(cd4);
    At = transpose(A);
       
    c_vector= [cd1t , cd2t ,cd3t, cd4t, At];
    y(i,:) = c_vector;
    
end


yt = transpose(y);
% wdt = y*yt; THIS IS NOT COVARIANCE

% FIND THE PCA OF Y(USING TRANSPOSE OF Y : TO OBTAIN 59 x 59)
[coeff,score,latent,tsquared,explained,mu]  = pca(yt);
% Explained feature describes that 1st column of PCA matrix accounts to
% 97.65% of variance. So only consider that.

% Projecting input data on 1st pca component
coeff_new = coeff(:,1:7);
x_pca = data * coeff_new;
events = [-1*ones(2048,1) ; ones(2048,1)];
X_train = [x_pca(1:3277,:)];
X_test = [x_pca(3278:4096,:)];
Y_train = [events(1:3277,:)];
Y_test = [events(3278:4096,:)];



% Mdl = fitcsvm(X_train,Y_train,'KernelFunction','gaussian','KernelScale','auto','ClassNames',{'-1','1'},'Standardize',true);
% CVMdl = crossval(Mdl,'KFold',3);
% 
% L = [];
% for i = 1:3
%     CompactSVMModel = CVMdl.Trained{i};
%     L(i) = loss(CompactSVMModel, X_test,Y_test);
% end
% KFold_loss = kfoldLoss(CVMdl);
% disp(min(L));
% disp(max(L));
% a = sum(L)/3;
% disp(a);
% disp((min(L)+max(L))/2);
% disp(KFold_loss);
% accuracy = (1-a)*100;
% disp(accuracy);