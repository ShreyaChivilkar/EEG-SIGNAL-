input = [ 0 0 1;
          0 1 1;
          1 0 1;
          1 1 1;
         ];
 correct_op = [ 0
                0 
                1
                1
                ];
 Weight = 2*rand(1,3)-1;
 for i = 1:1000
     Weight = SGD_method (Weight,input,correct_op);
 end
 save('Trained_network.mat')