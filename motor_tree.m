%%%%%%%%%%%%%%%%%%%%%%%%%% MOTOR IMAGERY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load the dataset

dataset_n = load('C:\Users\Saurabh\Desktop\Shreya\graz_data\dataset_BCIcomp1');

% Extract data from 4th to 9th second [(1152/9)x 3 = 384] excluding these
% samples.

j = 1;
for i = 1:140
    data_n(:,:,i) = dataset_n.x_train(385:1152,:,i);
    
end

% Applying wavelet transform 
for k = 1:140
 for i = 1:3
    x_n =  data_n(:,i,k);    
    [c_n,l_n] = wavedec(x_n,5,'db4');
    [cd1_n, cd2_n, cd3_n, cd4_n,cd5_n] = detcoef(c_n,l_n,[1,2,3,4,5]);
    A_n = appcoef(c_n,l_n,'db4',5);
    
    
    % Applying power spectral density on the required wavelet coefficients.
    
    psd_cd2_n = pwelch(cd2_n);
    psd_cd3_n = pwelch(cd3_n);
    psd_cd4_n = pwelch(cd4_n);
    psd_cd5_n = pwelch(cd5_n);
    psd_a5_n = pwelch(A_n);
    
    c_vector_n= [ psd_cd2_n',psd_cd3_n',psd_cd4_n',psd_cd5_n',psd_a5_n'];
 
    y_n(i,:) = c_vector_n;
 end
 y_new_n =[ y_n(1,:),y_n(2,:),y_n(3,:)];
 ip_n(k,:) = y_new_n;
end

% Applying pca on the transformed data

[coeff_n,score_n,latent_n,tsquared_n,explained_n,mu_n]  = pca(ip_n);

a_n = coeff_n(:,1:71);

% Projecting transformed data on the selected principal components

x_input_n = ip_n * a_n;

% Obtaining the output from y_train

x_output_n = dataset_n.y_train(1:140,:);


% motor_tree1 = fitcknn(x_input_n,x_output_n);

motor_tree1 = fitcknn(x_input_n,x_output_n);
save('motor_tree1.mat');

Cmotor_tree = crossval(motor_tree1,'KFold',10);
c_out = kfoldPredict(Cmotor_tree); 
save('Cmotor_tree.mat');



% extracting data for testing
data_test_n = data_n(:,:,113:140);

for k = 1:28
 for i = 1:3
    x_t_n =  data_test_n(:,i,k);    
    [c_t_n,l_t_n] = wavedec(x_t_n,5,'db4');
    [cd1_t_n, cd2_t_n, cd3_t_n, cd4_t_n,cd5_t_n] = detcoef(c_t_n,l_t_n,[1,2,3,4,5]);
    A_t_n = appcoef(c_t_n,l_t_n,'db4',5);
    psd_cd2_t_n = pwelch(cd2_t_n);
    psd_cd3_t_n = pwelch(cd3_t_n);
    psd_cd4_t_n = pwelch(cd4_t_n);
    psd_cd5_t_n = pwelch(cd5_t_n);
    psd_a5_t_n = pwelch(A_t_n);
    
    c_vector_t_n= [ psd_cd2_t_n',psd_cd3_t_n',psd_cd4_t_n',psd_cd5_t_n',psd_a5_t_n'];
 
    y_t_n(i,:) = c_vector_t_n;
    

 end
y_new_t_n =[ y_t_n(1,:),y_t_n(2,:),y_t_n(3,:)];
 ip_t_n(k,:) = y_new_t_n;
end

x_test_n = ip_t_n *a_n;

y_test_n = dataset_n.y_train(113:140,:);

L_t =[];
for i = 1:10
    Compactmotor_tree = Cmotor_tree.Trained{i};
    L_t(i) = loss(Compactmotor_tree, x_test_n,y_test_n);
end
KFold_loss_t = kfoldLoss(Cmotor_tree);
disp(min(L_t));
disp(max(L_t));
a_t = sum(L_t)/10;
disp(a_t);
disp((min(L_t)+max(L_t))/2);
disp(KFold_loss_t);
disp('accuracy');
disp((1-KFold_loss_t)*100);
accuracy_model_t = (1-a_t)*100;
disp('accuracy_model_t');
disp(accuracy_model_t);


