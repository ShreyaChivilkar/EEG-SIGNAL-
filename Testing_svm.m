load('Mdl7.mat')

% CVMdl3 = crossval(Mdl3,'KFold',3);
% c_out = kfoldPredict(CVMdl3);

load('CVMdl7.mat');

% sv = Mdl4.SupportVectors;
% gscatter(X_train,Y_train)
% hold on
% plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
% legend('A','B','Support Vector')
% hold off

L = [];
for i = 1:3
    CompactSVMModel = CVMdl7.Trained{i};
    L(i) = loss(CompactSVMModel, X_test,Y_test);
end

KFold_loss = kfoldLoss(CVMdl7);
disp(min(L));
disp(max(L));
a = sum(L)/3;
disp(a);
disp((min(L)+max(L))/2);
disp(KFold_loss);
accuracy = (1-a)*100;
disp(accuracy);



%%%%%%%%%%%%%%%%%TESTING%%%%%%%%%%%%%%%%%%

data_asc_T = load('C:\Users\Saurabh\Desktop\Shreya\BCICIV_1_asc\BCICIV_calib_ds1a_cnt.txt','ascii');
data_T = [data_asc_T(:,11), data_asc_T(:,15), data_asc_T(:,29), data_asc_T(:,27), data_asc_T(:,31),...
          data_asc_T(:,26), data_asc_T(:,32), data_asc_T(:,25), data_asc_T(:,33), data_asc_T(:,36), data_asc_T(:,39)];

bpFilt = designfilt('bandpassiir', 'FilterOrder', 6, ...
                    'HalfPowerFrequency1',8, 'HalfPowerFrequency2', ...
                    30, 'SampleRate', 100, 'DesignMethod', 'butter');
data_filt_T = filtfilt(bpFilt,data_T);

data_ascm_T = load('C:\Users\Saurabh\Desktop\Shreya\BCICIV_1_asc\BCICIV_calib_ds1a_mrk.txt') ;

% Select required input according to the markers cue position.

j = 1;
for i = 1:200
    pos = data_ascm_T(i,1);
    
    ip(j:j+400-1,:) = data_filt_T(pos:pos+400-1,:);
   
    j=j+400;
end 

for i = 1:11
    x =  ip(:,i);    
    [c,l] = wavedec(x,4,'db1');
    [cd1, cd2, cd3, cd4] = detcoef(c,l,[1,2,3,4]);
    A = appcoef(c,l,'db1',4);
    c_vector= [cd1' , cd2' ,cd3', cd4', A'];
    y(i,:) = c_vector;

end

% Apply PCA
[coeff,score_p,latent,tsquared,explained,mu]  = pca(y');

coeff_new = coeff(:,1:10);

% Projecting the data on the pca coefficients.

 input = ip * coeff_new;
 
[label, score] = predict(CompactSVMModel,input);

for i = 1:80000
    if (label(i,1)== 1)
        disp('Class A')
    else 
        disp('Class B')
end 
end 

