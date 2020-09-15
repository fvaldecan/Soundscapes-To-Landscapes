function [pcaModel,eigenVectors,meanA] = pcaTrain(trainBirds,type)
%PCA Model
if(type == "Features")
    A = [trainBirds.Features];
end
if(type == "Spectograms")
    A = [trainBirds.Spectogram];
end
N = size(A,2);
meanA = mean(A,2);
A = A - repmat(meanA,1,N);
L = A'*A; % C = A*A' would be too big
[V,~] = eig(L);
V = A*V; % if v is an eigvec of L, then Av is an eigvec of C with the same eigenvalue
% multiplying v by Av destroys its unit-normality property, so we divide it
% by its magnitude as below
for i=1:N, V(:,i) = V(:,i)/norm(V(:,i)); end; 

% matlab stores eigenvectors in increasing order of eigenvalues. To make it convenient take
% first k eigenvectors, we will reverse the order of the columns of V
V = V(:,end:-1:1);
eigenVectors = V';
% compute eigencoefficients
eigcoeffs_training = eigenVectors*A;
pcaModel = eigcoeffs_training;
end