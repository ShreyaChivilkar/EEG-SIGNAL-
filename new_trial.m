
data_asc1 = load('C:\Users\DELL\Documents\NIC\20190620162034_Shreya.easy') ;
data_asc2 = load('C:\Users\DELL\Documents\NIC\20190620164119_Hrishikesh.easy') ;
data_asc3 = load('C:\Users\DELL\Documents\NIC\20190620172025_Gurkirat.easy') ;

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
    j = j+1250+500;
    
end
