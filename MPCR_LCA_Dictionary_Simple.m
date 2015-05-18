%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
%
% Machine Perception and Cognitive Robotics Laboratory
%
%     Center for Complex Systems and Brain Sciences
%
%              Florida Atlantic University
%
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
%
% Locally Competitive Algorithms Demonstration
% Using natural images data, see:
% Rozell, Christopher J., et al.
% "Sparse coding via thresholding and
% local competition in neural circuits."
% Neural computation 20.10 (2008): 2526-2563.
%
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function HahnLCA_Dictionary_Simple3

clear all
close all
clc

load('IMAGES.mat')

I=IMAGES;

patch_size=400;
neurons=256;
batch_size=1000;

% k=0.1; %not needed with cubic sparsity function 

W = randn(patch_size, neurons); %Initialize Random Dictionary

for j=1:10000

    X=create_batch(I,patch_size,batch_size);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    W = W*diag(1./sqrt(sum(W.^2,1))); %Normalize Colummns
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    a = W'*X; %Feature Activations
    
    a = a*diag(1./sqrt(sum(a.^2,1))); %Normalize Colummns

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     a = a.*(abs(a) > k); % Hard Threshold

%     a = sign(a).*((abs(a)-k)+abs((abs(a)-k)))/2; %Soft Threshold

    a=0.5*a.^3;   %Cubic function acts as threshold
    
    W = W + ((X-W*a)*a'); %Update Dictionary
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    imagesc(filterplot(W))
    colormap(gray)
    drawnow()

end


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I=create_batch(Images,patch_size,batch_size)

[imsize, imsize, num_Images] = size(Images);

border=10;
patch_side = sqrt(patch_size);

I = zeros(patch_size,batch_size);

im_num= ceil(num_Images * rand());

for i=1:batch_size
    
    row = border + ceil((imsize-patch_side-2*border) * rand());
    col = border + ceil((imsize-patch_side-2*border) * rand());
    
    I(:,i) = reshape(Images(row:row+patch_side-1, col:col+patch_side-1, im_num),[patch_size, 1]);
    
    
end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [D] = filterplot(X)

X=X';

[m n] = size(X);


w = round(sqrt(n));
h = (n / w);

c = floor(sqrt(m));
r = ceil(m / c);

p = 1;

D = - ones(p + r * (h + p),p + c * (w + p));

k = 1;
for j = 1:r
    for i = 1:c
        D(p + (j - 1) * (h + p) + (1:h), p + (i - 1) * (w + p) + (1:w)) = reshape(X(k, :), [h, w]) / max(abs(X(k, :)));
        k = k + 1;
    end
    
end

end



