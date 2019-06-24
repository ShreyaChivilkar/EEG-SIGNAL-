load('Trained_network.mat');


clc
clear
data_asc_s = load('/home/namrata/Downloads/EEG-SIGNAL--master/20190620162034_Shreya.easy') ;
% data_asc2 = load('/home/namrata/Downloads/EEG-SIGNAL--master/20190620164119_Hrishikesh.easy') ;
% data_asc3 = load('/home/namrata/Downloads/EEG-SIGNAL--master/20190620172025_Gurkirat.easy') ;

data_ascn_s = data_asc_s(15001:75000,1:8);
% data_ascn2 = data_asc2(15001:75000,1:8);
% data_ascn3 = data_asc3(15001:75000,1:8);

% data_ascn1t = data_ascn1';
% data_ascn2t = data_ascn2';
% data_ascn3t = data_ascn3';

% Concatenating data of all the users

% data_asc = [data_ascn1; data_ascn2; data_ascn3];

% Extracting data of duration 2.5 - 3.5 seconds

j = 1251;
for i = 1:500:15000
    data_s(i:i+500-1,:) = data_ascn_s(j:j+500-1,:);
    j = j+1250+250+500;
end

% Applying  discrete wavelet transform on the columns

for i = 1:8
    x_s =  data_s(:,i);    
    [c_s,l_s] = wavedec(x_s,4,'db1');
    [cd1_s, cd2_s, cd3_s, cd4_s] = detcoef(c_s,l_s,[1,2,3,4]);
    A = appcoef(c_s,l_s,'db1',4);

% Transpose is used inorder to matcch the dimensions

    cd1t_s = transpose(cd1_s);
    cd2t_s = transpose(cd2_s);
    cd3t_s = transpose(cd3_s);
    cd4t_s = transpose(cd4_s);
    At_s = transpose(A);
       
    c_vector_s = [cd1t_s , cd2t_s ,cd3t_s, cd4t_s, At_s];

% Stored the coefficients of DWT in a new matrix 
    y_s(i,:) = c_vector_s;
    
end

yt_s = y_s';

% Applying PCA to the coefficients of DWT

[coeff_s,score_s,latent_s,tsquared_s,explained_s,mu_s]  = pca(yt_s);

% Explained feature indicates 99% variation is stored in 1st 4 columns 

coeff_new_s = coeff_s(:,1:4);

% Projecting the data on new bases (PCA as axis)

x_pca_s = data_s * coeff_new_s;

% j = 1251;
% for i = 1:500:15000
%     test_vector(i:i+500-1,:) = data_asc(j:j+500-1,:);
%     j = j+1250+250+500;
%     
% end

input_s = x_pca_s;
 n = 15000;
 for k = 1:n;
    trans_ip_s = input_s(k,:)';
    weighted_sum = Weight * trans_ip_s;
    op = Sigmoid(weighted_sum)
    if (op>0.5)
        disp('Happy :)')
    else
        disp('Sad :(')
    end
%       op = logsig(weighted_sum);
 end
