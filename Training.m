% Classification using Neural Network
 

clc
clear

%Load Calibration data 

data_asc = load('C:\Users\Saurabh\Desktop\Shreya\BCICIV_1_asc\BCICIV_calib_ds1a_cnt.txt','ascii') ;

% Convert the data into required microvolt format after selecting required  electrodes

data =0.1 * double( [ data_asc(:,6), data_asc(:,11), data_asc(:,14), data_asc(:,26), data_asc(:,27),...
          data_asc(:,28), data_asc(:,29), data_asc(:,30), data_asc(:,31), data_asc(:,32), ...
          data_asc(:,43), data_asc(:,47), data_asc(:,50), data_asc(:,54)]);
      
% Normalizing Data
 
an = max(abs(data));
datan = (data)/max(an);
      
 % Load markers file  
 
data_ascm = load('C:\Users\Saurabh\Desktop\Shreya\BCICIV_1_asc\BCICIV_calib_ds1a_mrk.txt') ;

% Select required input according to the markers cue position.

j = 1;
for i = 1:200
    pos = data_ascm(i,1);
    
    ip_mat(j:j+400-1,:) = datan(pos:pos+400-1,:);
   
    j=j+400;
end 


% According to the cue postion create the output matrix

j = 1;
for i = 1:200
   for k = 1:400
    op_mat(j,1) =  data_ascm(i,2);
    j = j+1;
   end
end

% Apply coloumn wise DWT

for i = 1:14
    x =  ip_mat(:,i);    
    [c,l] = wavedec(x,4,'db1');
    [cd1, cd2, cd3, cd4] = detcoef(c,l,[1,2,3,4]);
    A = appcoef(c,l,'db1',4);
    c_vector= [cd1' , cd2' ,cd3', cd4', A'];
    y(i,:) = c_vector;

end

% Apply PCA
[coeff,score_p,latent,tsquared,explained,mu]  = pca(y');

% Explained indicates maximum variance obtained in 1st 8 coloumns.
% Extracting data of the required columns only.

coeff_new = coeff(:,1:10);

% Projecting the data on the pca coefficients.

x_pca = ip_mat * coeff_new;

input = x_pca;
correct_op = op_mat;

Weight = rand(1,10);
 for i = 1:1000
     Weight = NN_method(Weight,input,correct_op);
 end
save('Trained_network.mat')


