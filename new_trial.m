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
data_asc = [data_ascn1; data_ascn2; data_ascn3];

j = 1251;
for i = 1:500:45000
    data(i:i+500-1,:) = data_asc(j:j+500-1,:);
    j = j+1250+250+500;
end
for i = 1:8
    x =  data(:,i);    
    [c,l] = wavedec(x,4,'db1');
    [cd1, cd2, cd3, cd4] = detcoef(c,l,[1,2,3,4]);
    A = appcoef(c,l,'db1',4);
    cd1t = transpose(cd1);
    cd2t = transpose(cd2);
    cd3t = transpose(cd3);
    cd4t = transpose(cd4);
    At = transpose(A);
       
    c_vector= [cd1t , cd2t ,cd3t, cd4t, At];
    y(i,:) = c_vector;
    
end
yt = y';
[coeff,score,latent,tsquared,explained,mu]  = pca(yt);
coeff_new = coeff(:,1:4);
x_pca = data * coeff_new;


target = [1,-1,-1,-1,1,1,-1,1,1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1];
j = 1;
for i = 1:90
    for k = 1:500
        ans(j) = target(i);
        j= j+1;
    end
    
end

targett = target';
anst = ans';
X_train = [x_pca(1:36000,:)];
X_test = [x_pca(36001:45000,:)];
Y_train = [anst(1:36000,1)];
Y_test = [anst(36001:45000,1)];



Mdl = fitcsvm(X_train,Y_train,'KernelFunction','linear','KernelScale','auto','ClassNames',{'-1','1'},'Standardize',true);
CVMdl = crossval(Mdl,'KFold',3);
c_out = kfoldPredict(CVMdl);

L = [];
for i = 1:3
    CompactSVMModel = CVMdl.Trained{i};
    L(i) = loss(CompactSVMModel, X_test,Y_test);
end
KFold_loss = kfoldLoss(CVMdl);
disp(min(L));
disp(max(L));
a = sum(L)/3;
disp(a);
disp((min(L)+max(L))/2);
disp(KFold_loss);
accuracy = (1-a)*100;
disp(accuracy);
        
