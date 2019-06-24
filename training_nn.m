clc
clear
data_asc1 = load('/home/namrata/Downloads/EEG-SIGNAL--master/20190620162034_Shreya.easy') ;
data_asc2 = load('/home/namrata/Downloads/EEG-SIGNAL--master/20190620164119_Hrishikesh.easy') ;
data_asc3 = load('/home/namrata/Downloads/EEG-SIGNAL--master/20190620172025_Gurkirat.easy') ;

data_ascn1 = data_asc1(15001:75000,1:8);
data_ascn2 = data_asc2(15001:75000,1:8);
data_ascn3 = data_asc3(15001:75000,1:8);

% data_ascn1t = data_ascn1';
% data_ascn2t = data_ascn2';
% data_ascn3t = data_ascn3';

% Concatenating data of all the users

data_asc = [data_ascn1; data_ascn2; data_ascn3];

% Extracting data of duration 2.5 - 3.5 seconds

j = 1251;
for i = 1:500:45000
    data(i:i+500-1,:) = data_asc(j:j+500-1,:);
    j = j+1250+250+500;
end

% Applying  discrete wavelet transform on the columns

for i = 1:8
    x =  data(:,i);    
    [c,l] = wavedec(x,4,'db1');
    [cd1, cd2, cd3, cd4] = detcoef(c,l,[1,2,3,4]);
    A = appcoef(c,l,'db1',4);

% Transpose is used inorder to matcch the dimensions

    cd1t = transpose(cd1);
    cd2t = transpose(cd2);
    cd3t = transpose(cd3);
    cd4t = transpose(cd4);
    At = transpose(A);
       
    c_vector= [cd1t , cd2t ,cd3t, cd4t, At];

% Stored the coefficients of DWT in a new matrix 
    y(i,:) = c_vector;
    
end

yt = y';

% Applying PCA to the coefficients of DWT

[coeff,score,latent,tsquared,explained,mu]  = pca(yt);

% Explained feature indicates 99% variation is stored in 1st 4 columns 

coeff_new = coeff(:,1:4);

% Projecting the data on new bases (PCA as axis)

x_pca_nn = data * coeff_new;


target = [1,-1,-1,-1,1,1,-1,1,1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1];
j = 1;

% Storing target data into ans matrix. 30 images were shown each corresponding to either 1 or -1, to 3 subjects. Total 90 images.
% Images are given as output for 45000 time instance.

for i = 1:90
    for k = 1:500
        ans(j) = target(i);
        j= j+1;
    end
    
end




targett = target';
anst = ans'; 


input = x_pca_nn;
correct_op = anst;

Weight = 2*rand(1,4)-1;
 for i = 1:1000
     Weight = NN_method(Weight,input,correct_op);
 end
save('Trained_network.mat')
