
clear all, close all,

C = 3;
M = 4;
N = 10000;
n = 3;

gmmParameters.numberOfClasses = C;
gmmParameters.component2label = [1,2,3,3];
gmmParameters.priors = [0.3,0.3,0.2,0.2];

gmmParameters.meanVectors = ...
    [0 2.5 1.25 1.25;
    0 0   2.2  0.7;
    0 0   0    2.2];

for m = 1:M
    gmmParameters.covMatrices(:,:,m) = eye(n);
end

[x,componentLabels] = generateDataFromGMM(N,gmmParameters);
labels = zeros(1,N);

for m = 1:M
    indm = find(componentLabels == m);
    labels(1,indm) = ...
        repmat(gmmParameters.component2label(m),1,length(indm));
end

save('Q2data.mat','x','labels','gmmParameters');


size(x) %[output:676ccccb]

sum(labels==1) %[output:5b91f0dd]
sum(labels==2) %[output:640f11b2]
sum(labels==3) %[output:1d5c15d0]


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright"}
%---
%[output:676ccccb]
%   data: {"dataType":"matrix","outputData":{"columns":2,"name":"ans","rows":1,"type":"double","value":[["3","10000"]]}}
%---
%[output:5b91f0dd]
%   data: {"dataType":"textualVariable","outputData":{"name":"ans","value":"3005"}}
%---
%[output:640f11b2]
%   data: {"dataType":"textualVariable","outputData":{"name":"ans","value":"2992"}}
%---
%[output:1d5c15d0]
%   data: {"dataType":"textualVariable","outputData":{"name":"ans","value":"4003"}}
%---
