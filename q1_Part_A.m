
% Question 1 - Part A

clear all, close all,
rng(1);

C = 2;
M = 2;
N = 10000;
n = 4;

lossMatrix = ones(C,C)-eye(C);


gmmParameters.component2label = [1,2];
gmmParameters.priors = [0.35,0.65];

gmmParameters.meanVectors = [-0.5  1;
                            -0.5  1;
                            -0.5  1;
                            -0.5  1];

gmmParameters.numberOfClasses = C;

gmmParameters.covMatrices(:,:,1) = (1/4)* ...
    [2   -0.5   0.3   0;
    -0.5  1    -0.5   0;
     0.3 -0.5   1     0;
     0     0     0     2];

gmmParameters.covMatrices(:,:,2) = ...
    [1    0.3  -0.2   0;
     0.3  2     0.3   0;
    -0.2  0.3   1     0;
     0    0     0     3];



% AI-assisted: Use mvnrnd to generate random samples

labels = double(rand(1,N) >= gmmParameters.priors(1));

N0 = length(find(labels==0));
N1 = length(find(labels==1));

x0 = mvnrnd(gmmParameters.meanVectors(:,1)', ...
            gmmParameters.covMatrices(:,:,1),N0)';

x1 = mvnrnd(gmmParameters.meanVectors(:,2)', ...
            gmmParameters.covMatrices(:,:,2),N1)';

x = zeros(n,N);
x(:,labels==0) = x0;
x(:,labels==1) = x1;

fprintf('Class 0 samples: %d\n',N0); %[output:0ed23d5c]
fprintf('Class 1 samples: %d\n',N1); %[output:404bd7b3]

save('Q1data.mat','x','labels');

% AI-assisted: Use mvnpdf for probability density 

mu1 = gmmParameters.meanVectors(:,1);
mu2 = gmmParameters.meanVectors(:,2);

S1 = gmmParameters.covMatrices(:,:,1);
S2 = gmmParameters.covMatrices(:,:,2);

g1 = mvnpdf(x',mu1',S1)';
g2 = mvnpdf(x',mu2',S2)';

discriminantScoresERM = log(g2./g1);

[Pfp,Ptp,Perror,thresholdList] = ...
    ROCcurve(discriminantScoresERM,labels);

% AI-assisted
Pfn = 1 - Ptp;

figure(1), clf, %[output:2fcc47ec]
plot(Pfp,Ptp,'*m'), %[output:2fcc47ec]
xlabel('P(False+)'), %[output:2fcc47ec]
ylabel('P(True+)'), %[output:2fcc47ec]
title('ROC Curve for ERM Discriminant Scores'), %[output:2fcc47ec]
grid on, %[output:2fcc47ec]


save('Q1_A2_results.mat', ...
    'thresholdList','Pfp','Ptp','Pfn','Perror');

saveas(gcf,'Q1_A2_ROC.png'); %[output:2fcc47ec]


% AI-assisted: Calculate probability of error using class priors
PerrorWeighted = gmmParameters.priors(1)*Pfp + ...
    gmmParameters.priors(2)*Pfn;
[minPerror,indexMin] = min(PerrorWeighted);

% AI-assisted
tauEmpirical = thresholdList(indexMin);
gammaEmpirical = exp(tauEmpirical);

gammaTheoretical = gmmParameters.priors(1)/gmmParameters.priors(2);

% AI- assisted
PfpMin = Pfp(indexMin);
PtpMin = Ptp(indexMin);


fprintf('Empirical optimal gamma: %.6f\n',gammaEmpirical); %[output:8fd8f633]
fprintf('Theoretical optimal gamma: %.6f\n',gammaTheoretical); %[output:2e1783dd]
fprintf('Minimum P(error): %.6f\n',minPerror); %[output:9b535d2f]
fprintf('P(False Positive): %.6f\n',PfpMin); %[output:40d665d5]
fprintf('P(True Positive): %.6f\n',PtpMin); %[output:4135dfe7]


figure(1), hold on, %[output:2fcc47ec]
plot(PfpMin,PtpMin,'ro','MarkerSize',12,'LineWidth',2), %[output:2fcc47ec]
legend('ROC Curve','Minimum P(error)','Location','southeast'), %[output:2fcc47ec]
grid on, %[output:2fcc47ec]

saveas(gcf,'Q1_A3_ROC.png'); %[output:2fcc47ec]


function [Pfp,Ptp,Perror,thresholdList] = ROCcurve(discriminantScores,labels)

[sortedScores,ind] = sort(discriminantScores,'ascend');

thresholdList = [min(sortedScores)-eps, ...
    (sortedScores(1:end-1)+sortedScores(2:end))/2, ...
    max(sortedScores)+eps];

for i = 1:length(thresholdList)

    tau = thresholdList(i);
    decisions = (discriminantScores >= tau);

    Ptn(i) = length(find(decisions==0 & labels==0)) ...
        /length(find(labels==0));

    Pfp(i) = length(find(decisions==1 & labels==0)) ...
        /length(find(labels==0));

    Ptp(i) = length(find(decisions==1 & labels==1)) ...
        /length(find(labels==1));

    Perror(i) = sum(decisions~=labels)/length(labels);

end

end


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":64}
%---
%[output:0ed23d5c]
%   data: {"dataType":"text","outputData":{"text":"Class 0 samples: 3528\n","truncated":false}}
%---
%[output:404bd7b3]
%   data: {"dataType":"text","outputData":{"text":"Class 1 samples: 6472\n","truncated":false}}
%---
%[output:8fd8f633]
%   data: {"dataType":"text","outputData":{"text":"Empirical optimal gamma: 0.612892\n","truncated":false}}
%---
%[output:2e1783dd]
%   data: {"dataType":"text","outputData":{"text":"Theoretical optimal gamma: 0.538462\n","truncated":false}}
%---
%[output:9b535d2f]
%   data: {"dataType":"text","outputData":{"text":"Minimum P(error): 0.024924\n","truncated":false}}
%---
%[output:40d665d5]
%   data: {"dataType":"text","outputData":{"text":"P(False Positive): 0.019274\n","truncated":false}}
%---
%[output:4135dfe7]
%   data: {"dataType":"text","outputData":{"text":"P(True Positive): 0.972033\n","truncated":false}}
%---
%[output:2fcc47ec]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAQAElEQVR4AeydB5zcRfXA395fCGpAREoIATZAIApCqEYQ2QsQQIqQ0AVyoQmhhSIm1AtI71IEAXNRCRCIhKI0DRdCCy2U0EuWEkpAQIgCSvnPd+7eZva3v9\/u\/rbd7t7c597OzJv+fvNm3rQ3LV\/7P08BT4FICrSI\/\/MU8BSIpIBnkEjSeA9PARHPIL4VeArkoYBnkDzE8V6eAlVkEE9cT4HGp4BnkMb\/hr4GVaSAZ5AqEtcn3fgU8AzS+N\/Q16CKFPAMUkXi+qQbnwKNySCNT3dfgwahgGeQBvlQvpg9QwHPID1Dd59rg1CgxxjkvffekyeeeEKmT58uzz33XI+S6\/PPP5e5c+fGAuJood955528cdPptFDfL774QqPkmG+++WZOGv\/6179ywoEgrWB5o8IS3oV58+bl5KNpaTm\/\/vprN0qWnTo88sgjokB6WQHqxPHxxx9nykhZP\/nkk5JKVnMGWbBggRxxxBHyox\/9SEaMGCH777+\/nHrqqSUVvlKRnn32Wdl8881jwV\/\/+tdM9kcddVTeuMOGDbP1XX311YWw77\/\/fiauWnbaaaecNE477TT1zjIPOOCAnLAnnXRSVpgoR1tbW05crbuWc+2117bf6KWXXspJ5tNPP5XddtstA3\/5y19ywtQD4rHHHsuUkfLyjUspV80ZpKOjQ2699dZSylqTOMVm8tVXXxUbNCvctGnTZJtttpFXX301Cx\/meOihh3LQ\/\/73v+Wpp57KwX\/55Zc5uFIR5ME32mqrreRvf\/tbqck0RbyaMwjDnUs5eq9NN93URTW9\/Z\/\/\/Kccd9xxBeuJ2DV\/\/vyscHPmzMlyV9tx6KGHypNPPhmZTZ8+fSL96smj1HLGZhCG2P\/+978l1\/2tt97KxB06dKhceeWVcvDBB2dwQQuyZLCRBMPkczNXyCdTh8VFBJw1a5bkg2233TYsqsU988wzovD000\/L3\/\/+d2lvb7d++vPwww8Lcxd1R5nM01y\/2bNnu86y7AMGDMiUk\/L+4x\/\/kMsvv1zAuwnzjdS9+OKLywsvvJABxD31C5q0E8TJuKMb8xwgmF4xbv3em222WaaMlHfIkCEFo9NOADdghkGQJffZZx9RcHsNKnnCCSdY2XXNNdeUwYMHyy677CKnn366fPDBB256kXaG7HHjxmU1CiaF4C644IKseMi+v\/zlL2XjjTcWKgYj\/fCHP5Q999xT7rjjDgn+MSppuUeNGiUQ95prrrF1+f73vy9vv\/12MEpeN41gmWWWkXzwzW9+MzSNb3\/724KfAu5VVlnFlmXHHXfMivPyyy9nudXxve99T63y+OOPZ+xYkK0xATcc7rhAr6rlxBw4cKAMHz5cbr75ZnHTRszSyTgd5H777ScKN910U1a2+F988cXy05\/+1LaTjTbaSAYNGmTbDswXtZjw4YcfyimnnCLQiLkawPc\/5phjchZxivnezDm0jJjuQhBpggNuueUWobM6\/vjjhbIy\/8KfzpGKtfADXH\/99XLfffdlgAKCJzIy8+TJk4XVDnAAH+qqq66yFXrjjTdA5YX7779fpkyZIsi3GpAeFBwfQHHI6Mi+d999dxYzEQ+ZfMyYMTJhwgTLBBqH\/LXsM2fOlPPOO09OPPFEWxcNUw\/m0ksvnVWM5ZZbLsutjvXWW0+tQp3VwbxH3T\/4wQ8sI6pfJc3vfve7svPOO2clOWPGDOum81FaYyrj4Mlov9122wkdHuIhOAXaztlnny2\/+MUvhIUaxWMyKsKYzE\/d+RXtg46b0fr2228nqIVivjcrfZRPgbLZyOaHutxzzz0CTJw4UXbffXe59tprBdGXdkaeLB6xEGMZBK6GK01c+7\/rrrta4hP4kEMOsRGtR8gPhGA1irAh3hnUt771LaE3zSC6LeCWWmop62LkYJXHOvL8TJo0SWDoqCBXXHFFlFdReAjE6BgFjIZFJeQEmj9\/ftboRw+92mqrOSEWWhnBYAAwNBh6ZeyMOEpnVgHBVQvoTd20ixmFYQwYQeMhqq2\/\/vrqtCY9O43SOswPIlGhNmaCCWFgGOxBKOd7u5KSmy50Puyww7puFD7wwAOun+ywww7WzQgBV1mH+Rk9erQQ9t577xWYyKDsP2HYz7COiB+WIZHH9cMT7Gc\/+5mA08Z+zjnngM4AcwFkYxiH4TnjYSxnnnmmuL2CQWX99+vXz45uiGp9+\/bN8ivk4CNS9yhgThGVBoRlYusCYgNiIp2JxqPXSiQS6swxXQaASQjgfsxgw8O\/kqCdlqZZDIPcdtttGtyKK7STG264wY7kMIt6um0FUdht+Ihmd955p9BJtba2ahRrPvjgg9YM+ynnezM\/RPK49NJLc5JuATOje\/jETs+mH4fhCRxAT\/\/rX\/9aKAiVxQ5e4cYbb1RrSSYbOW7Do0HBIMjG\/\/d\/\/2dl4yOPPDKTNg0R8S+DcCzIkRD5\/PPPF8q5xBJLOL7VtyIyuqANXHNG3qVu6g4zhwwZkkEjguBArMUEmJNhVgtaWmzTyCSPeJdxRFjoKNWLb3PRRRcJZUe0pJe\/+uqrBXBX8NxvTtxzzz3XzlmoP50gOAW3\/orDLOd7M59m\/rrCCivY5ffgyNlCxZG1yAhA9qRBYn\/++ecxLNAg29ra7ESZyTJDnvXo\/glOJrvRRRuvvfZaVlhk2SyEcSCnGiPzH7WXsMceewhiSiZgTMs666yTqSd1DQKTx5hJZgWn0Xz22WdZuKCDBqI4FX8ZvcHRia244opYqwbpdDorbTrGLESIY8MNN8zCwiAjR44UGuEZZ5whrEYSBtCASAdqp6HDTOpmkYRvrPCb3\/xGvbLMcr43CwhuYslk0nVKC+IEjV+xKl7Ro7t4\/JkgugBOIRhW8cWaTLzcsMsuu6zrtHYahrV0\/wSZqhstUbK9+mfMdFoknRazzJVBYdl+++2FjxEF7MwSLgpoGAr0iIx8btkZXRDfouKDhwE0Dr0xYoiKaDrCE65a8OKLL2Yl3c+IrFmIEAcLI1rmoDfSyPjx460koHs5zD\/cUWfllVcORivKXfT3DkltkUUWycJ+4xvfyLiRmlqQExWz6qqrCsuiuBFtMBVYAmQyng80bClmUOYNm1\/AtG7a9DCuW+3BtBRvmcHs5Mvo0SJmSTMDI0ZkglgLsqhZKbPhLaL4H4gKgylALyZ7l112WVYidDRZiBCHMgKdD7K8BnFXubpwlf1lpYoVHjdVbRcuLmhfa6217MoQS\/fuXNMNB6MfeOCBwh6JSirqH1zdUnwhM\/J7F4pYhH\/LXXfdlQnmTrzhJBhGPSEQvWE+0LClmMHe49FHH81JBtHERbK\/4Loj7TAFEz6YAubAnU5HBhf2J9rbFzIR4aNDF+WD+OAGdHtOF+\/a11133YyTFSJ1uHjFVdJk9EOy0DSZcxZaFKBxIwqy30D52BZgL4GJL2K7poUJk7zyyitCGyNtcABxMRXoJNlv45waEOxkNFw1zRZ3Asl6s5vZGmuskXEiFjzxxBMZN+vfyOJMFoExZn8i41mChT2BgTTg7rjXXXeduJMyiHrhhRd2+3YZwUbXhXV+02kRGAOm6Ox0PPJb5\/TpIzPNRqCFd9+VmaZuM03ZZpqRhdUOQMWE\/Ckt9GVTbqFLhLmf6w6zu\/MQ1z+qd3bDFGNHxKFRKrDpx+objdqNv9dee0lw0u76Y6cdIXoqMOIxwrOHxv4HYibhFGj82N1vyDdmHww8wHI6bYDVO6DWiy2UoYUfgFWj\/v37Y83AQQcdlLFjQVzgaAFE5PwUFWL4B9jaJ0ypkEgk7IqTG5\/eg\/wQUbbccktRGZwwrDwEJ1TgLcybJ2Y3sWsE6Oy0qDg\/08yy8Cgjc2eBKd8oswHJTj2AGBEnTcIifmEC2kCwR0EYI9CTB5ktKn4hPPSkU1Q4+uijcw4nshS\/7777FkoqI5prQE4ic0qbPStGpD\/+8Y\/qZU3EMSzsZmMqsA\/Gt2WRhjmN4qEd5VR3rcwMgwSHQQpAJYKcz3kdRhP8FSAijVndpZowAbK7G5\/8WGWDCRXPSBNcRVM\/a\/72tyLt7dZaTz\/uwgMiFh1MvvIxD3R7WMJusMEGGDUBviuiHaJQoQzZfWcS7oZjQ3CCmcvBINRX\/WACGjxuxLHgN2dC74p4hGMUIg\/stYQMg9A4wzKm9\/7973+fc4BNw7LSw9mbQkOwhs9nJhIJgZj0PFHhmCexIcXwHRXG7DJFelmPZFLELFnLxIliZpXSEphA2zCFfpxTtcGVkKiozONcP3pXdbtpuA0yuC5PgwqLo7hyTRiSzVW+OftIiwRWeYLf2S03o0F7e3voiQnKxQoX\/kFxHCY89thjCZIDxJk6dardo8jxjIFYdNFFY4TuCrrYYotJi64x59s32GKLLezNPzZ1IBwyKhtxnJJkjyCRiN4R7spq4S+NW\/O85JJLFno4Ns7rsMxIfkzM+FAcKmPXnc0jelYnuL149er06fKqWSZ8de5cGRh2NyKZFIEp7rlHxISxzIE7lZJ1jAihZcqYpDdqlJBeGNzGvg3zG1MQGrrGo4wGFfpPfTUcJpuYGpA9DnAAdVQ8m2rgFNy9IEZXxZO2xslncsZN44SZzAEoF988rFHR87vxXDEc5kE8Yk+M70VnR8\/PkjZSACMD\/oRzy4ibdDg1QTzmmpycYIWVJW63UyAeor5bBiQK8C60mm\/jhnHTYDFB\/Vh0cuPR4asf4TIjiBsozE6vxqoRhGPixQaL23uExSkHp\/ltvfXWwlENxD0+TmSakyaJdHaGeyeTYhmCEcMwRHigADaZFDn55C5miorT2Sl2rhOIGuo06wUywfj0Aljk9EVkralryfaPby87P7OzDJs5TL4\/5fvS58w+hl7RNPjm2d+08XaYvYMMf3C4DLh6gCROMZ1vXJpBa5NNJf6LZpBKZFa1NNJpETO0h6V\/85JL5m\/kYZHAmSRtg55kGCVlRh1JSegf+U6AUYzvhDwwyfi1e5Ba0EAq99ccDDLabPyF0GSqWY06YcCAEJ8ASpnBbeDBBi35mMRELObDB7L1zipRgG\/XYdJ2wThL+W98BunsFOnszKn7R0suKb9aZpkcfA4C5oCgYQ08J3AUk5A\/XyMngkfkoUDVvPiW9JkulPh5Gp9BzDJiDqGTSXk1SssHDAGxFJQ5chKJQph5SahX14WiUC+P7FkKwDBtpRWhsRkknRbp7MyteSolH62zTi7eBBe3V8EO8XJD5sGkjF+bgeA\/HEcGQbx39ygF+L5RfVoRBWtsBunsDK+iWZ7N8aDtwhARUXLC50WMivCtSOIRaXt0bAqUyRzk19gMwtIutXDBiFeSSrkYkYoyh5g\/0k8aM\/jvxawgRXrMzSBPP8a3dyFmgRqbQdLp3Oq2teXimGdUvHOHSYJZIWYFcd7dAxQQ4VMMFJEgmAVHgy36v3EZJJ02I4OBYFU32yyIqZK7VvlUqfjNnizi1VxTySDEnI80LoOYuvfsf5iIRYlCmBa0h9pSAAZBas2qdwAAEABJREFUauAzuRCzFE3PIP3\/29+MNDGpUlRwqF5UQB+opyjAogyiVhn5NzWDDPhigPz8o593yaNlECk8qh8pwulSZ9gymaRxGSSZDP8S6XQGP\/KTkTJm\/piMuzaWiHLVJnOfSxgFymCSXAYJyyCA4\/IS107\/8Ic\/CMeFOZLNHQ6OXOtVWS7+B6JV3plM5qY5o2upddiwYXLE2CNy\/SuG6conO7mQ8mQH8K4aUOChrR8SmWgyciFkcdOEKPhfNIOghYKz+pzn5w76KLMZx9l57mvAFNwe494GzLLLLrsI5++5bPVQyBsXBUtVbIBUKjdkOp2LqwqGGWAwYc8gQYr0iBtmCEKJBSnIIKiD5\/L8JptsImPHjrVqJN280JfE3WmUg3H7S\/0YZbgkw4UqLvIHNZJouLLMsCXdzk6RdLqsZAtHJn2TT07AEIYNhuHDscJSQ\/js15\/JgqMXiNQwz57Ka+paU2XeN+ZJpf7yMgg3q7jdh5pM7hQPGDBA9DomN73w5yYctwS5p84NLHAzjJiDuMVowyUn8CNHjhRUblaq4DadtjZr5PyEHWDMCVQkgiyCDSsZsdvUZvZGgmGDboZ91uJrCJ\/++lP55OhPRGqYZ0\/lddF3LyrywxYXLJJBUAnJ7UFEJBQKo5Wis7PTah4B76qIDGaFVkAu\/Le3t9u3F8466yzhWiQq5oNhy3anUrlJdHSIdHbm4uNiSDrYoEeZ0SNt0g+mlUoZuddAoUYYjOfddU2BSAb56quvpLW1VdCyzbsNP\/nJT4S7w3Frw\/1x5iQoqGNUiRu\/YHgzFwoNM5qli1Cf4pGmvWcFTqfFamXMQnY7uJ7bbfVG81AgkkFQGocm7iFDhhSs7f\/+9z8B8gVEzSSjSr4wJfm1tYmkUrlR02mR9mz1+bmBYmDSabH3zzs7cyMlkyKpVC7eYxqeApEMUmzNeHwHZgIK6XkqNs3Y4SJ7bxpzBZgknRaZNEmkoyO8aCiDCPfx2AanQNkMUhf1T6VEzHwnvCwwCUc6McND5MWm02JkTZGo9MGnUnmT8J6NS4HmYBDozyiSSmELAdPIhZGE1SfsIUFyUCZchwk\/0DBXOp3jaxGplFkZYlZuXf6nCSnQLAzS9WlQCpdKddlDf9sN1jR4AUzjF0QmRhYF3OBhJhMmTXgJ\/0smRcgv3Ndjm4QCsRmEyTi76i4oLcL82GhU\/5qYNNpUqkBWaeNP42elC2ZQwA0ehjFBov5TKbHaGaP8Pb5pKBCLQT788ENhMs6uuYI+8gJF0PCueDXfeustvGoLMEl7e+XzTCZF2tv9yFF5ytZtirEYpCYHECtFKuYk7XNNaikDFfhPJs1GoNk1JN0KJOeTaAwKxGIQ1M9zvsoFfcKZ6nJg0fXDzlkt\/HoGTKMWlL0BJTJKMtnFGCi8TqV6pho+1x6jQCwGQaE0KvxdWH311TOFR\/xy\/bCzQZgJ0GMWGjZMwohiRCTBLRF\/hiHEbD6KGS3aTJy5Jk5bm\/i\/3kmBWAzS+CSi8bMsaxq+fG2qAxgGEAV1G+YQwxTJfIwk\/q8XUKCXMUjYF00apIKx+n9PAYcCnkEcYnirp0CQAmUzSN++fYXJOOBemApm5N2eAo1IgbIZhEk4k3Ggmi9ONSJxC5bZB6h7CpTMII899picd955cuCBB8ruu+8uV155pa3sRRddJLxdaB3+x1OgwSkQm0G+\/PJLOfTQQ4VLUDzmyUObXL9VpoBBeMOQW4QNtbHY4B\/SF786FIjNIOecc45w\/1yLw1VatWPqxuAVV1whkydPBuXBU6BhKRCLQVDIwDPQ1BblDc8995zVi4VbgaeJR3dfd4WZPv74Y\/XypqdAw1EgFoPw\/jU15H46b2n36dMHZxZwB338+PHCYUVU\/yCCZQXwjppQwGdSGQrEYpA5c+bYXIcOHWrNqB+OpPz4xz+23vPmzbOm\/\/EUaEQKxGIQ9jyoJHdBMPPBZ599Zr3RjmIt\/sdToAEpEItB2Ougjtdcc42grAF7GHBvBDU\/+KGmFNODp0AjUiAWg2y88cbCbjlaFtG4+MQTT4i7lIs4hZ7evffeWwiDVsX111+\/Eeniy+wpYCkQi0G4D8LeBzGfffZZGTFihKB3FzcK5jbddFNBeTV+Yv7YE\/nOd75jbP6\/iSjQq6oSi0GgzEYbbSRckkJhNe4wQIfvVVddJTxBEObvcZ4CjUKB2AxCxTbccEPhKQT09ba3twtPIbD3MW7cOGGf5M477\/TMAaE8NDwFSmIQao2eXvZD0OB+8skny4knnmjPZaHYmr0QwnjwFGh0CsRikE8\/\/VTmz58fCzi71ehE8uXvvRSIxSC33367sEkYB3pMX2\/v\/aYNXPP6K3osBql28b\/++mt577335JNPPql2Vj59T4GiKBCLQVpbW+3knAm6CzfddJOw\/PurX\/1KeGxHc+b5hBVWWEGdkSaMwYOga6+9tqCIbp111hGeSrjggguy9lkiE+j2uPjii4UFBAUWDLq9vOEpUBIFWlZZZRUpFtj022GHHSQIO+20kxxyyCHC6d177703U5D99tuvqLRXXXVV4UFQDjdq5Oeff15o8KgVKrZ8MBQblApn9jlTVhlo6lcqTDJxY9Cn2HLWOhzfjTcma51vpfJzO11tH7UyW7iz4WGyvbvi6VB\/dOBty1oxQ1g+LXEm3D7s0NiLFJ5m5dGMTeewhhsbV2KEWHOQEvPw0TwFGpYCnkEa9tP5gteCAp5BakFln0fDUqBHGWTBggWC+qCZM2fKm2++KSz3FktJ9kseeOABIe7cuXOlmEtcUWlTjtmzZ8v06dOFe\/a4o8J6fO+iQIvw2JLE+7vxxhvzLt\/uuOOOcuaZZ0qUwgYaIvsc7HugPojDjizl4T7iiCPyXsbijjv3Utgv2WuvvYS4m2++ubCUyX5MnBuMr7zyilBW8mW1ZP\/995dtt91WcKPzy112jkchH7qSFODOEZ1WIZg6dWols7Vp5Y4gPNNnvTI\/ORb3fBXKGVzgktRTTz1lT\/XS6N5\/\/\/2s+H\/+85+Fhsg+R79+\/WT77beX0aNH29UhGuStt95qcapny4187rnn2gORHF\/huD3MsfPOOwt5Evfoo4\/O0bLixnft3HjccssthbJSDtKhHLpqwsZnW1ub8KycG8\/be4YCSAyFgI1qJJFKljCbQXi\/crRJvggmMaFs74uOLBeefvpp4cwWNw\/poe\/hOTQCGyDcSSedZGxiNxZnzJghXKriJDB7EHfccYewaUglaayu2EQ6l112mY17+umny2233SacIj777LMF8QgTz8svv1weeeQRrJEAoQ866CDrzyiEqiLiUw42OjklgCfin2qMxO2hZyiAEhD0rxUC1FJpB1epki5kEJij3SQLtBmzjH8e0uFKLkloY0X0YacbHD0\/vX1Qly+75n\/6058IIowSiEw4GLFgCuzs2O++++5YMwABGQE4fg8SRsSMgksuucR6caTllFNOkeDx\/LXWWkvojQikw\/bnn39uGY85CngXwFFPRAHw3MnHDSMyr3rooYessj0YDjyiAuFcYB6FH3EUT3pPPvmkINIysnKlWf28WRsKdDGIyxy8L1OBvFVnFiMJydE4GFGw08gxwwBxR\/21ob\/88suicffdd9+waBZ3xhln2OMpzG8sIuKH68F4waSYYYB4xVEX9H\/hz2OkXC\/WERCcwqmnnmqvHsPU4BjRCMvow5xmzz33lEmTJomGQxM+4Vw47LDDbBqInuBhhj322EM4xnPssccKczOuNI8bN66sBQnS9lA8BVqkCsyBiHTttdfaUnBFF0s6ncawk+mll17a2qN+NM5LL71kg7zxxhvWZN7BvXjrCPnhYCQNksOKId4WRe\/OfAVHPo0rjCqkNXz4cIKWBFw7psEzujHqsSBBQsr42AFEA+7x05mgTwzREuagU2GOxs1NRFHmelOmTLH3\/onnofoUaBHmG4hUo0SENuyC5P9DdKDnU0CpNatRAEzCZJynoUnl9ddfxxDkSGuxP+E\/KkfSIyNmvPbaazbgSiutZM1yfrQcpFELhRLc32d+w0igIxtL0x988AFFsKAMw0iDuIhmGOiHCAhjIDrCKKhbYkGCEdAVxWwi\/qcqFOgSsWCSgSb9IIA36Kh\/GjDiggIfmg+r4RGXuJqLm4aOSc+MmQ9U6RyNgQajcRZbbLF80Yry07TorYuKUEYgFhzc0YzRT5mEpW5NWudaP\/\/5zy3qvvvusyZMxS1OBerPEjeejz76KIaHKlOgi0GYmOs7lq7JyJKnAHxsPpTCrFmz7H2RI4880sbiCLv22Mlk0uJcBrKIkB\/kb9A0MEyNq6IWuDCgIbF5yGQ3zB\/ciiuuiGH1dhHeOiJ+SIf0CoVjASIsCUbSIJ59F3Cs9GEywWeCzj4Ox8PBIZZhohBjzTXXFBfuvvtuvPLuFdkA\/qciFGgRmAPoFBHasAuS\/2\/RRReVpZZaKgPLLLOMsAKEyMUHJzbMg7nyyitjCG+JhK3iWM\/uH5Z7sa622moYogyCTJ7vtuHNN98sLNuyyWcjhvwwgqgIx5J0SBCLYlGAiTbpWUSeH+0EgkF4fSuIg2kYGVm2RjslGmAIs+uuu2JY0NEWvWN0NmGA+GUD+5+qUqBFmHtMNHnE2P8woQv+0+sR6KOPPsIQPihiDRNkncBbj8APowcyNmhtnIhqADjkcMwwQHYHTyPEjAKOoOPHZiBmGLC0Ch75H6bCDgRHE9yImvgVA3Qqygzsv2hdt95660x0lrtxsKNPZ+MC5aHDQFwjjIfqUqBLxEKUqjCTaKOCIagCbpYrsbMcG9bQEaFYvSEMx0eGDBmCVZjHMNHFgakyO26AHre9vd3uitM72waIRwSwx0E4JsvsgwR3y2m0PABE9DFjxmBI\/\/79rclqkzuKsVJlPWL8cCOT4Gx0InIyUiy++OKgLCgDM5Kyj2KR5oeJOWqWWPJlbmZQ\/r\/KFOhiEDJxmaQTRHmgHxz5WlNi0kljwM2uNUc9jjnmGGETkM1DVrxoMDQQnnAjnAI9pzIP+xfbbbedHH\/88cK+ABocWQolLKs+hZaREQXJk\/AdHR1CvkcddZTALMwRKBN+9NyUBTv7OoyC2CkH+fCIkG5+gi8WSIfVPO08lCYaH1owYrJKyBuQN9xwg30Dkjkfcci3UB01LW+WR4GFDEI6MMk9xpIykOefHh1vNbEHYfnll7eoadOmSTqdtnZ6Pc5TnX\/++fb8FHI+vTW9ML05vToNgsbO3MZGcn5owIw+hKMnR1RjXwCmYs7DQUaYxYkSaWXZlOMtLAQgIlFOmIWzWYiCKJxA9ncToNzMX8gbBmEOAxPBVG44tYfNQdSPfRHsMAIHL7ErMNrCFKyAIYaxWUm9uWvP0jnMrGG9WV0KZDMIeRVgDoKw4cXmFo0ddxgwWhAGSCaTWUFoUDSu+++\/XyZOnCg0xs7OTisiMSLASFkRuh00OCbO7FTTcGAkjoKgZZ4GpatA3T7uI1wAABAASURBVMELGowOHFqkHByihOFYtWL1qrW1NSc+vT5ntWBmlrZZjoWJYBzqCfMQCSbFTcPGHQYHHHCAEIb8qFcwDJuezKmgCx0IzEy9UQ4ePKITjOvdlaNAi2S3XanlH6MM4g2NkU3ARCJRVPYwEI0VsWvdddeVJZZYoqh4YYESiYRQDvYX6Mnp0cPCuTgaL2+l5BtB3fCl2hOJhEAXGA5mrsXGZqllbdZ4uSNIs9a0gvXySfUeCngG6T3f2te0BAp4BimBaD5K76FAixg5VwYOFJkwQSSd7j019zX1FCiCAi02TDotYjbaLKPALGZFyeL9T91TgM1Vd6+pngvMRidnz+q5jMGydTGIi02nRVji7OhwsVl2dnJZUgU4wZvl6ThYDiUMwLKmerFLDi5KqYOGCzPLiRuWXk\/jaDDQohuylGFwX4UldT3YGCwr9\/3ZPGQvKehXj272m7hjw35VPZYvrEy5DKKhRo8WiRhJuAKrwTggqPagyV5BEIc7eLQDXLFQTtxi86hlOJeWgwcPFoWBZiRn15wDmtywRPt9sFzsQxGGnfegXz26YXg6OG5lUu56LGOwTNEMQkhGkggmwRvg+LV7NgkcwB3uKAbhICCjS9++fQkaC8qJGyujGgdmk5HRWIGNUEaXww8\/3JaEqwMcjLQO8zNnzhzhFAE764107ISzbYwkjfI0RYvZyhZpazMkj\/hn8h7hRS+HF0e3MV1gh5heQsO4fpzwRbzSjTZkaHavuWrKlVguE7FLzlkkt1GQRlRcwpEfwzeHGXkqjvAADY1d6TDFB+xmkzdHzwmrQPnAI+MrDgUK7GbjRp5GtCE\/8gVHp0Acyg5NgmkSJg5w\/oujLBpHj+zgVg0vemQFnAKHNykrnUlYnQkHg2ldENWgD\/fuoT91oH4clOQbwLTEARjBwXHHnhMQM2bMyLkjny8NNnYZJbnvD41Js56hxTLHxIkic7kpFVJURhAgxEtPpfIRgt46emgY159eD1lUCQShOUIC0Tmqjn+7WTTgCioHA90RCr+wuBwaZAjnLBeHGdl5ppdiOCc8RzSYO6H4AObV8tDQyPuZZ55RlDVhKvDukXgaK7q+uG\/CrjuHGsmPfGEUGitxKDtvo2yzzTby9ttv2\/RK\/eEYCmfPiE8HgkkD5KQv58iYu4BT4LoANOOoD6entc4c4dH4hMVNXThVjQ4A6MO9FpiGOgQVThCHtPkmwATTcZ522mlWpxlHh1w9ZvnSIB3ELExGScx6hoUiVjIp4uiwyir0pElZTnVwX6Ffv35CRd3ekt6c3pVDfxy40\/CFTA7k8dHRbcX5JuJzeJARoVBczitRHkQRJq6ER7sjZ6w4NYy8rniOzONfKsA03N\/gRDCKJEgHRqHX5jg66YNHlFA1RoSJC\/Tg9N46QuloTA9OWqhXwlSAAWAO5i0cxmQk41AlPfaUKeHKHjhVTVk5Iay3LUkPenKzURVOcGuSMKQNHTkbBm1hDsKxCMN3J65CMA3Faz06OzsVVbfmQgahiKmUSFsbtmzo7Mx2d7s4EwWBcMIkmACH\/zD5SPSA2IsBjoEjFqBJhHT5eMTjA2DmAxiRE7n0bui9gnEJz4jD3Q6OlNPjgYPpEEOwlwL0yow8jBgws6bBiNVuRj4aFXdOwAdHJnBhwKjAaKfAiV16dhRhEJ7GqWfO6KHBcU4LUwExiZPN0BHG4Kwa34BRglGITgvRScNjMmJDNw5WcsYMnIKrcIKbj5xihtkuvPBCqwmT82u8KkY+5Bu2YOOmoekqgyBKK65ezWwGoZSbbcZvNqTT2W7HRW+CU7URYqfXw0TEwCwW6H1hOg0\/aNAga2WeYC15fhApXG+NSwNQPIf9mAzjpkFilgLcRdF4epEKN8+cYQLau9NwcBcCRgl6ZAUaLcfbiQdDIjZiB5gzYOo1ZuwAp4sxoQW9uUI+ZQ90KC7NiQ8wktPpYAeYl2CmUilxw9MBgsMv2JEF0yAMoOWmzox64OoVchkkmQwvazodiucOOo2OY+B8OD4KHxfxaL311guNE4UM3gHhXgRhETUw88Fyyy2X5a0fkXJkeRTpyJcnF640GV1owK3lxR4XGPFYEXSBBs+dGRhE60O6yjiuSAReGyijGFeeXSBdwriiMO4NNtgAIwfcjgVP7aRcpgEP6BXhdDqNMwPBNNSD78yIhjtYHnD1BLkMEla6ZFIkmQzzsTjEFyyIWSzfYkdEonfBXiyUc8+BFZ9i8ykmHCs6UeHC8tIPHhWnEB4GoMd1gdEpkYi+AvCf\/\/wnK1kVG\/ke3FMJA8QvN1IiEZ5+8NtpTx\/2jXQhgobvph1MQ\/3ofBg9cIfREny9QC6DmNWJuIXbaqutbBRkUF29UtHLejTAj+ri0qKyoqP2ejN1VNReXcunPTmLFay4ucA8IZlMSqnKHnS1TBcINE\/M2bNnY9i7K9ZS4Ofdd9\/NhNCr2RlEnVmyGSSdFunszC2ikTtzkQsxXB5i4gXxWPLlAwZ7qoWh68tGo6FErEBhAkxkmdhir0dQ2gYZhKVtyssSML00doD6ML8Limr4FQssGBCWyTr7PdgBRCQV37bYYgtQBYHlYgIhmicS4SMY\/vUACxkknRZ7BiusVKNGhWGzcIhUimAVx5XNFV+P5pAhQ2yx2LhCywhLt2g4VDnfetbZD\/fvKRLzE0wFjpwwl6GjYtmZa8jsZzCaI9KwElbqrjt7SaxgkSfqmFhCBtDiD61YSmc+qmXJZ6oqWW5K5gtXD34t0tEhZrdH7EnedDq3TMmkSIERhEh8BEyAPQJMANkaMx8oM6mZL2zQT+Ooqf7qDsrBKkMnEl09F6KHHudgz4K1e0ZE9mE0raCZSHTFDeJdt+ar5XD9wuxarjC\/II6GypyHSTn7E+rPIgFMwUSa+SBLt+wt0YBZ7mbpWMOqqeVUt5pBPG42ctloZS+k3SxnAyz9ombpt7\/9rUbNmMTJOByLKiWH9g66Lq0tljk6OsILl0yKPYoS8GWfgeEdYqkXk0twgIoA+NHbgaMnww2w4gVO1\/XRVoJbRQTCAMlk0io2YC0dN1BsXPIjTZXLiQvQcMC7PenYsWOF\/QrmUHx8dvYZEQnHZiPxAM0b5W+4FQiHEgp1Y7LSBV7FD3BhQK9LOMoV5h+Go9Nhp55RQfecNBx7GeyHdHZ2CszOhh77JuyUu0zIA0Tkq99A49Org4e5FKcmdULzCwsx0IjvQtrQyJ1s50uDoyrsdSFepYroeDXvnjIXilhhJWhrE2mASoQVPS6O3pcjI6VOYuPmV254RBtGEcScYFqJRPWUPSQSCYEJW1tbhZGKvaVg\/vnczI8Y0eiUYPR8YfHraQhnkGRSxAyhcnKFXtPp6Vo2Yf705uPHj7fHfIKjV71Wl6P9iGJIGIzQ9VpOt1y5DJJMinAmyzOHS6e6tHPUBTEXkaUuCxgoFLvxnMZmIaTYuVkgiZo7WySZFEkmRdrauhiDU73JZM0L4jOMTwEaGRNnTtbGj137GMwxYRIWQWqfe2k5tthj7jAFR95TqdJS8bE8BZqUArkiVpNW1FfLUyBAgaKcnkGKIpMP1Fsp0MIZGw+rZGkT8fSoH3pwr6UnmbOFjaQo4IIRm3jlALvTUelXA7\/33nsL9x8g6sgFI2Xy25NLh61M3MmND1zk4oJXNehdizS5oSk99NfCykIUcH2TczflAJeIotKvNH7GjBkCQ+rJ3EGLDJKhnw0tHfqZuEMbHzjNwN7D0AauSw\/xhzTNHOTFF18UrtZCSPRIoUSAA3u4PXgKlEqB0hik1NyqGE83y7gLwUUh99xRFbP1STc5BZqGQRhB+FacKmYDDbsHT4FyKdB0DIK8XS5RfHxPAaVAUzAIiiLQQUWlGuU0LmX1UP8UaAoGcRUs1Psd555oEjxUylXZnsi70fOsOwYphaDuBSaOVJeSRlgcrq5OnTpVGh0effRRQTt8o9ejmPKHfcdycE3BIK66GdX3Ww5RNO68N+cJO7mNDrNmzbJVavR6FFN+W9EK\/jQFg3CzTmny8ssvq1VklLHOFXnnoHeMJcZ\/uwlr4o18eqRwzdbDvQ1Fh5EjR5oPWJn\/pmAQSIEOWkxVe4pduNZi4PF\/Pi45f23dGJih25oxVjY2E8\/8+v9eToGmYRC9wgmD8DaHflfewXCfO7B4mEJvEzPKTLRY\/+MpkEOBpmEQrp6ySUgN0fjBdVT0QPE+yef\/\/Rx0Fxjm+OL4LwRlal0I88to0oBMgnI46uG+n2Jq06v\/K02TpmEQNGSgy0qZ5OGHHxZVucPxE1rN5+M+l5O\/OllQBaRyKs8LTJs2TaSBmIRGwKoU9eIhHw4iUm8eEVL9vFLEH4rm0EwSBXo6oYik6iYIK3bQBCWAlShU0zAIxGC5F51dqBHlGQFU4qDvau+\/7y1fXPmF7PPqPva0L2GXXHJJDOGtEBSq2Ud6lEkwrW99\/nD8m4eC0ItFCVH1ihI5tEOecMIJAgOBLwTot0IFTxQElWMXSq+n\/Vni52h\/JcvRVAyihGGzkMdd0NyHvivwU741RVAYgB0ioiQO+37774chvL+BwjSpc+agVw+eWqZeKG8Q8zdlyhS7b2Osef8RzQiATmWuCYRBIyhXYMTk+bvf\/e53gsrYOAr4qH8haEoGCas0Iwp45iUoWU4MTIiY+cjeJ+wtPEEm5k810xtr3f5HnVpmDoZ6UQrOS1KY+UAf9qHuvDMSBq62xHxp9aQf8y+uNfDSFWpQK12WXsEgDL2IIBBPn2rALmYli5O\/MAzu0o9jELs2wAhCTsw5KDt2BX2whhMAhTZM33jjDRtttdVWs2aj\/iAtIG6eeuqpAiA1VLIuvYJB9IEXCId6UUwX0I+LW7WOY69XUAYJO7XMi1Jabl77UnuYqXWdP3++jBs3Tlgmh+mOOeYY4WWrsDj1iGNxhmfkUMUKuDSoRHl7BYOk02lLK3bcwzSOq35ZJr3ItDZwHf4UOrXsKqJm4p2vCnPRhWYCXHvttcK8hcUKRlnEs3322Ud4TNR49\/r\/XsEgKm8vu+yyoR9cJ\/J40ggx6xEKnVp2mX\/BggV5q2AXJEwI3hNBoQMrWszB9EQCczaeUjBBevV\/r2AQln\/5yqjexwyCu5zJKBP0rxe31oPyMK\/CdMGtnzuauGHUftBBB1mZnT0g3nNBYzurVhwI1DnZRLRtaoReavYKBkETOt836ulnFCrjz35CcOILvkfBybzQqWWeQ9PghS6OwRTI7GGjKqcQSAeRy31uDVxvg17BIPpYDnOMd955J+cb64QVhXE5nnWEcEe3rFPL3WXUt\/9wLrfcchihwDwLUTKq8bv5uGJbaGJNjuwVDDJo0CBhdOBbhm0k6aahihaEq1fQOQKHMoNl5HEacCz3uo0cnAvTp08XVns22GADgVlcP+xsPGIOHjxYWCXC3luhVzAIH5mHRfnIPBfmjiJMRHWDqREYhOVY6gGDBE8tM7HGT8NgZ4GCIyiA7o3cJZNaAAAJvElEQVSsu+66eAkjKme6rKP757nnnsvoF9tpp526sb3X6BUMwufdd999ZeDAgbZRcAyFHXXWz\/UtPs434U\/YegZ2zNmvoIzBU8s0+BEjRtg9DfwBlnM5xAjgD445me6602HACKiX5eAmr9kSDlqw3Ev43gy9hkFoFKgl5bFRPjgnfdlxRhTh\/XAdYfCrZ2A0jDq1DOOwm+yWP5FIuM6M\/dhjjxXqDYLDnR0dHcKohJsRaNq0adIIR00obxhUarGl1zAIROzfv789An\/\/\/ffLVVddJWyKzZ49W7ShEKYRgOXesFPL4Nw9HerC0Qv2PIDll18elAUYjXrDHChDuPTSS4UTzbhhQI5w2IAN9oNEQF3PPffcipS8VzGIUoyGMmzYMBkyZEhDT0JpxIiLMEGQMbSuhUzSYE6yzTbbCPdKcBeK05v8eyWD9KYP7OtaHgU8g5RHPx+7ySngGaTJP3APVa9psvUM0jSf0lekGhTwDFINqvo0m4YCnkGa5lP6ilSDAp5BqkFVn2bTUMAzSNN8yt5SkdrW0zNIbentc2swCngGabAP5otbWwp4BqktvX1uDUYBzyAFPhgXitBCGAXuNdcCSckHH3wgt956q3DnQsNyRyMq7TB8sWpFNX3X1Lw4zu7i683OfXsUSDz22GM9XjTPIAU+we233y4oQ44CDvqha+uAAw6QF154IW9qp59+uj057N6N32677fKmH8wXTYJ5M8njyf0O0gteksoTpUe8uObLaetddtlF3GvEVS9MSAaeQUKIEhdFj8xVXk7Eoj4nLD7XWDlej8I3TuCGhak2jtGw2nlUKn0UipPWhAkTMHoMPIPEIP3kyZMFBlCAKa6++mpBN6wmw4eFYdSNicig6vgPP\/xwUDnAEwQwUCHggldO5CZEcHOS5x1QTM0d+p6qomeQGJRHRQ76oxS4ltra2iqo7tSGj0ZD1OW4yaLzFxzhubvh+qmdNLmfUggQPzROs5tcKaaOp512mnz11VdYaw6eQSpEcno8TYp74GrH5D44Jg\/WJBLhV2DxjwuMVNddd51wl5xrsrvttpud43AzEL9i00MFEPMS7ulzbZdyMl\/KN0l++umnhRuMjJ40ZOyIkcXkyRyMm5wvvfRS3uDDhw8XRkzoyaidN3CVPD2DVIiw7h1o9y43jeCVV16xuWy++ebWrMQPqkV5Xu64446zd8nRrUsDZZWMt07wYxWsUF6EQeM9GtK5p89IR2PUSfKVV16ZlQQ9ub7FwdVcRCC0q2CHQVH+UGiuA+PxwheMmJV4wAEdufkJmnph1hoqxSC1Lnfd5YcSaC2U+6TAAw88YNHo5UKMso4K\/NDD07OS1PHHHy+oL6IX32OPPUAJfsHGbT0CPzAGqoHoqUnn+uuvt\/f10a1F0DPOOEPQAI8dQLkDb3FgJwyqSrn\/zcgDDn9eucJeCWDVjXQQU2FO7LUEzyAxqM27h+5LTCwBIx8zMujDNkOHDhV03GqyKIjArh8aexigRYTl4iggD43HXgiiFW4m\/4g4rI7RSCkPdvxQwIAZBTQ47Zn32msvIR0WC+i1WXxAsTVxGVkw2cdRcRH1pOjuRZEdqoZgTlUlhLb4Qs8vkF4xwDI64RAZCy2jE67S4BkkBkURZ0aPHi0KhxxyiNCQ6K1JZsCAAUKPjF1BNbKjUUVxUSaNIArcXhwl1fT2AEwRTA9NkuBICzMKYDT1Q7skcwN1sxiAIjpGFF2Wvu2226xeMcKMHz9eEons+RR7QfgBd911F0bZgLomTUQf\/VF3LUzPIDGojJhEr+oCK1NoZGR5F71SQf2+7777rs2hkLYQem6eIYgCRBmbkPlB7Q+9PQATMA\/gjUJGDzbXdHQxQfP+wwSMAARC2yQjDxN+4r\/++utCXSgXdSQM6nQwEceYd9xyyy3iwoMPPoi3BZ13sXs\/Z84ccUF1IbPp6eKx6wNBNhHzs+SSS5rfrn9GsC5b7X49g8SgNb0pcwoX2AvhUVAaVt++fXNSY9kXpD7Sgz0MmJ+gcT0KEHvceBxXYZILnpUkNtQYzZgAu+EK2ZnLuKMQTM5ImUqlZMsttxReCtaRRhkEphw7dqyMDQHNjzDYEUtZMHABRsaPMC4ee1tbG14ZQH+XOtwRTnHVNhuAQapNgtqk775BUm6O9O677rqr3bQkLXp+9mGYLHOG6cgjjwRdlM4vRgPmD2zGtbe3C6MhOBJgFAB3+eWX48wSqcgjH3CqgEju6h7uQuAyRDBsoVE4GL4Sbs8glaBinjQQy\/COc6iR8PnA3edAKyIrWPTmTJZZINBRK18a+PH8Ae83stSbTCaFs1qMho8\/\/rgw\/9Cy6wodIhfxAOZhhx12mASByTtqXGFawjHCMfK4QB74Iaq6eOz33nsvXhlgj0Yd7vsoiqu26RmkyhRmZ5wsKike6NsgLAroKg95AOxBMBqoHTMKEH822WQTe1jSLd8iiywiTMwPPvhgG5VlYFa83OVr9BpbT+eHxQpW60gz2NCdYLGs+rgRkTyDQIUmg4022sjWKO7cwEaK+FlxxRWtDw3XXcpllKJRgycAjRozCgYPHpzxYifdDc9K2axZs6w\/Z6IQlZjQ66iCLl93KZdVNo7cEAERbbPNNsNaNriT9jXWWKPs9OIm4EeQuBSLGZ5VIKIgz7u9NLhSQeV74vN0AcdMAEYTFg101YldcSa9UUfGWUJl\/4N0aPAcvecZaFbH1ltvPavoG78xY8ZgCO8eMifBAWO2trZasQyxin0advLxY68EJsFeLnAkhTQQ2Qo9K0e4SkPvZpCY1Mw3gYxKipectLHkG0XipM1m4nnnnZfJkmMmAPlw1IM5ijImoo4uEITlwTMIjAwkBkNxmpglXFaYSO+ss84SzkThD\/B+iM5PCHPfffcJohp2GBNGY6JP2ChgeTnKL4hnYxYccxnMWoNnkAIUp\/EweQRWWmmlAqFzvZHnebwHHybTmC7QgEmbB3xcfCE7IwcMxzIseySMHDAJIwI9PUvSrGgBOrlmt568mFhr+ixNE5\/zV6xmcbQE5mNizikA9lU0rJrMT\/AjbVbOiM9iATvu7uim4YMmIw\/lYLk86Oe62TlnpALHxB+z1uAZpAYU5zVZsqEBRYk7+McFRA6Oz8PE9N6JRPbONitaQDE9Nvsw7Idw4BDmY7INo0WVic1K0mbljPwR75inRIUvBQ+TEo8jLIiD2GsNnkFqQHHukegEllOyNciy4bNgZ5\/RETGPhYeeqpBnkBpRnn0DVo04EqLns2qUdUNmw6lgCs5hTEZK7D0BnkGqRPVgssxFOMaB7F3MwcVg\/N7mZsSFVj0191B6\/z8AAAD\/\/1FgRHUAAAAGSURBVAMA8yc8nJX5JNIAAAAASUVORK5CYII=","height":420,"width":560}}
%---
