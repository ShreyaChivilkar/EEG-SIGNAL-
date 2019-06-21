function Weight = SGD_method (Weight,input,correct_op)
alpha = 0.9;

n =4;
for k = 1:n;
    trans_ip= input(k,:)';
    d = correct_op(k);
    weighted_sum = Weight*trans_ip;
  op = Sigmoid(weighted_sum);
%      op = logsig(weighted_sum);
    error = d -op;
    delta = op*(1-op)*error;
    dWeight = alpha *delta*trans_ip;
    for i=1:3
    Weight(i) = Weight(i)+dWeight(i);
%     Weight(2) = Weight(2)+dWeight(2);
%     Weight(3) = Weight(3)+dWeight(3);
end
end