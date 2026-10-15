

clear all, close all, clc

Xtrain = readmatrix('X_train.txt');
Xtest = readmatrix('X_test.txt');
ytrain = readmatrix('y_train.txt');
ytest = readmatrix('y_test.txt');
x = [Xtrain; Xtest]';
labels = [ytrain; ytest]';

[n,N] = size(x);
C = length(unique(labels));

for l = 1:C
    xi = x(:,labels==l);
    Nclass(l,1) = length(find(labels==l));
    gmmParameters.meanVectors(:,l) = mean(xi,2);
    gmmParameters.covMatrices(:,:,l) = cov(xi');
end
gmmParameters.priors = (Nclass/N)';

% AI-assisted: Regularize each covariance matrix
alpha = 0.01;
for l = 1:C
    S = gmmParameters.covMatrices(:,:,l);
    lambda = alpha*trace(S)/rank(S);
    gmmParameters.covMatrices(:,:,l) = S + lambda*eye(n);
end

% AI-assisted: Numerically stable Gaussian likelihoods for 561 features
for l = 1:C
    a = x-repmat(gmmParameters.meanVectors(:,l),1,N);
    R = chol(gmmParameters.covMatrices(:,:,l));
    b = R'\a;
    logpxgivenl(l,:) = -sum(log(diag(R)))-0.5*sum(b.^2,1);
end
logpxgivenl = logpxgivenl-repmat(max(logpxgivenl,[],1),C,1);
pxgivenl = exp(logpxgivenl);

px = gmmParameters.priors*pxgivenl;
classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,C,1);
lossMatrix = ones(C,C)-eye(C);
expectedRisks = lossMatrix*classPosteriors;
[~,decisions] = min(expectedRisks,[],1);

for d = 1:C
    for l = 1:C
        ind_dl = find(decisions==d & labels==l);
        ConfusionMatrix(d,l) = length(ind_dl)/length(find(labels==l));
    end
end

Perror = length(find(decisions~=labels))/N;
disp('Class sample counts:'); disp(Nclass');
disp('Class priors:'); disp(gmmParameters.priors);
disp('HAR confusion matrix (rows predicted, columns true):');
disp(ConfusionMatrix);
fprintf('HAR training error = %.6f\n',Perror);
fprintf('HAR training accuracy = %.2f%%\n',100*(1-Perror));

[Q,D,xzm,yzm] = performPCA(x);

figure(1), clf, hold on
for l = 1:C
    ind = find(labels==l);
    scatter(yzm(1,ind),yzm(2,ind),8,'filled');
end
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('HAR Dataset - PCA Projection');
legend({'Walking','Walking Upstairs','Walking Downstairs', ...
    'Sitting','Standing','Laying'},'Location','best');
grid on
hold off
exportgraphics(gca,'Q3_HAR_PCA.png');
