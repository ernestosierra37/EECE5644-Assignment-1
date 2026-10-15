clear all, close all,

load('Q2data.mat','x','labels','gmmParameters');

lossMatrix = [0 1 1;
    1 0 1;
    1 1 0];

[decisions,confusionMatrix] = ...
    performERMclassificationGMMdata(x,labels,gmmParameters,lossMatrix,0);

disp('Confusion Matrix:'); %[output:227caf5e]
disp(confusionMatrix); %[output:070cf052]

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright"}
%---
%[output:227caf5e]
%   data: {"dataType":"text","outputData":{"text":"Confusion Matrix:\n","truncated":false}}
%---
%[output:070cf052]
%   data: {"dataType":"text","outputData":{"text":"    0.8080    0.0846    0.0859\n    0.0792    0.8005    0.0874\n    0.1128    0.1150    0.8266\n\n","truncated":false}}
%---
