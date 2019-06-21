load('Trained_network.mat');
input = [ 0 0 1;
          0 1 1;
          1 0 1;
          1 1 1;
         ];
 n =4;
 for k = 1:n;
    trans_ip = input(k,:)';
    weighted_sum = Weight*trans_ip;
    op = Sigmoid(weighted_sum)
    if (op>0.5)
        disp('Happy :)')
    else
        disp('Sad :(')
    end
%       op = logsig(weighted_sum);
 end
 