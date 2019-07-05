load('motor_tree1.mat');
load('Cmotor_tree.mat');
j = 1;
for i = 1:140
    data_a_t(:,:,i) = dataset.x_test(385:1152,:,i);
    
end

% Applying wavelet transform 
for k = 1:140
 for i = 1:3
    x_a_t =  data_a_t(:,i,k);    
    [c_a_t,l_a_t] = wavedec(x_a_t,5,'db4');
    [cd1_a_t, cd2_a_t, cd3_a_t, cd4_a_t,cd5_a_t] = detcoef(c_a_t,l_a_t,[1,2,3,4,5]);
    A_a_t = appcoef(c_a_t,l_a_t,'db4',5);
    
    
    % Applying power spectral density on the required wavelet coefficients.
    
    psd_cd2_a_t = pwelch(cd2_a_t);
    psd_cd3_a_t = pwelch(cd3_a_t);
    psd_cd4_a_t = pwelch(cd4_a_t);
    psd_cd5_a_t = pwelch(cd5_a_t);
    psd_a5_a_t = pwelch(A_a_t);
    
    c_vector_a_t= [ psd_cd2_a_t',psd_cd3_a_t',psd_cd4_a_t',psd_cd5_a_t',psd_a5_a_t'];
 
    y_a_t(i,:) = c_vector_a_t;
 end
 y_new_a_t =[ y_a_t(1,:),y_a_t(2,:),y_a_t(3,:)];
 ip_a_t(k,:) = y_new_a_t;
end




% Projecting transformed data on the selected principal components

x_input_a_t = ip_a_t * a_n;


[label_t, score_a_t] = predict(Compactmotor_tree,x_input_a_t);
answer = load('labels_data_set_iii');
ans = answer.y_test;
countsum = 0;
for i = 1:140
    if (label_t(i,1)== ans(i,1))
               countsum = countsum+1;
    
    end 

    
end 
value = countsum/140*100;
disp(value);
