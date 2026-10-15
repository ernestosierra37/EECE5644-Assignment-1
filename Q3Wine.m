clear all, close all, clc


data = readmatrix('winequality-white.csv','Delimiter',';');
x = data(:,1:11)';
labels = data(:,12)';

% AI-assisted: Standardize features before Gaussian estimation
x = (x - mean(x,2))./std(x,0,2);

classes = unique(labels);
C = length(classes);
N = length(labels);
n = size(x,1);

for l = 1:C
    xi = x(:,labels==classes(l));

    mu(:,l) = mean(xi,2);
    Sigma(:,:,l) = cov(xi');

    Nclass(l,1) = size(xi,2);
end

classPriors = (Nclass/N)';

alpha = 0.01;

for l = 1:C
    lambda = alpha*trace(Sigma(:,:,l))/rank(Sigma(:,:,l));
    Sigma(:,:,l) = Sigma(:,:,l) + lambda*eye(n);
end


for l = 1:C
    pxgivenl(l,:) = evalGaussianPDF(x,mu(:,l),Sigma(:,:,l));
end

px = classPriors*pxgivenl;

classPosteriors = pxgivenl.*repmat(classPriors',1,N) ...
    ./repmat(px,C,1);

lossMatrix = ones(C,C)-eye(C);
expectedRisks = lossMatrix*classPosteriors;

[~,decisions] = min(expectedRisks,[],1);


predictedLabels = classes(decisions);

for d = 1:C
    for l = 1:C
        ind_dl = find(decisions==d & labels==classes(l));
        ConfusionMatrix(d,l) = length(ind_dl)/Nclass(l);
    end
end


% AI-assisted: Ensure predictions and labels have matching dimensions
Perror = sum(predictedLabels~=labels)/N;

disp('Wine quality classes:');
disp(classes');

disp('Class priors:');
disp(classPriors);

disp('Confusion matrix (rows predicted, columns true):');
disp(ConfusionMatrix);

fprintf('Training error = %.6f\n',Perror);

[Q,D,xzm,yzm] = performPCA(x);

% AI-assisted: Plot the first two principal components
figure(1), clf
hold on

for l = 1:C
    ind = find(labels==classes(l));
    scatter(yzm(1,ind),yzm(2,ind),12,'filled');
end

xlabel('First Principal Component');
ylabel('Second Principal Component');
title('Wine Quality Dataset - PCA Projection');
legend(string(classes),'Location','best');
grid on
hold off
exportgraphics(gca,'Q3_Wine_PCA.png');