load('motor.mat');
load('Cmotor.mat');
j = 1;
for i = 1:140
    data_a(:,:,i) = dataset.x_test(385:1152,:,i);
    
end

% Applying wavelet transform 
for k = 1:140
 for i = 1:3
    x_a =  data_a(:,i,k);    
    [c_a,l_a] = wavedec(x_a,5,'db4');
    [cd1_a, cd2_a, cd3_a, cd4_a,cd5_a] = detcoef(c_a,l_a,[1,2,3,4,5]);
    A_a = appcoef(c_a,l_a,'db4',5);
    
    
    % Applying power spectral density on the required wavelet coefficients.
    
    psd_cd2_a = pwelch(cd2_a);
    psd_cd3_a = pwelch(cd3_a);
    psd_cd4_a = pwelch(cd4_a);
    psd_cd5_a = pwelch(cd5_a);
    psd_a5_a = pwelch(A_a);
    
    c_vector_a= [ psd_cd2_a',psd_cd3_a',psd_cd4_a',psd_cd5_a',psd_a5_a'];
 
    y_a(i,:) = c_vector_a;
 end
 y_new_a =[ y_a(1,:),y_a(2,:),y_a(3,:)];
 ip_a(k,:) = y_new_a;
end




% Projecting transformed data on the selected principal components

x_input_a = ip_a * a;
answer_m = load('labels_data_set_iii');
ans_m = answer_m.y_test;

[label, score_a] = predict(Compactmotor,x_input_a);
count = 0;
for i = 1:140
    if (label(i,1)== ans_m(i,1))
               count = count+1;
    
end 
end 

value_m = count/140*100;
disp(value_m);
