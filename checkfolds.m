
clc; close all;
%%   This code checks for intersection among the train spilt and test split across the given fold.
%    Import the csv files in the table format.

%    naveenp@iisc.ac.in 

%%   Check Image  wise intersection

c = intersect(F2train,F2test);
if isempty(c)
    disp('Train and Test Split are images  wise separated')
end


%%   Check Patient wise Intersection

file = F2train;

for i = 1:height(file)
    
        if     file.Label(i)==0
               cls = 'CNV';
        elseif file.Label(i)==1
               cls = 'DME';
        elseif file.Label(i)==2
               cls = 'DRUSEN';
        else
               cls = 'NORMAL';     
        end
        
    trn(i,1) = sscanf(file.Name(i), strcat(cls,'-%d-'));
    trn(i,2) = file.Label(i);
end


file2 = F2test;

for i = 1:height(file2)
    
        if     file2.Label(i)==0
               cls = 'CNV';
        elseif file2.Label(i)==1
               cls = 'DME';
        elseif file2.Label(i)==2
               cls = 'DRUSEN';
        else
               cls = 'NORMAL';     
        end
        
    tst(i,1) = sscanf(file2.Name(i), strcat(cls,'-%d-'));
    tst(i,2) = file2.Label(i);
end

 c = intersect(trn,tst,'rows');
 if isempty(c)
    disp('Train and Test Split are Patient wise separated')
end
 