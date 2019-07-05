load('motor.mat');
load('Cmotor.mat');
L =[];
for i = 1:3
    Compactmotor = Cmotor.Trained{i};
    L(i) = loss(Compactmotor, x_test,y_test);
end
KFold_loss = kfoldLoss(Cmotor);
disp(min(L));
disp(max(L));
a = sum(L)/3;
disp(a);
disp((min(L)+max(L))/2);
disp(KFold_loss);
accuracy = (1-a)*100;
disp(accuracy);

[label, score] = predict(Compactmotor,x_test);
count = 0;
for i = 1:28
    if (label(i,1)== y_test(i,1))
               count = count+1;
    
end 
end 
