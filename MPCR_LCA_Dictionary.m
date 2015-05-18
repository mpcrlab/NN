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
% Locally Competitive Algorithms Demonstration
% Using natural images data, see:
% Rozell, Christopher J., et al.
% "Sparse coding via thresholding and
% local competition in neural circuits."
% Neural computation 20.10 (2008): 2526-2563.
%
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function MPCR_LCA_Dictionary

clear all
close all
clc
tic
load('IMAGES.mat')

I=IMAGES;
% [imsize, imsize, num_Images] = size(I);

% for i =1:size(I,3)
%     imagesc(I(:,:,i))
%     pause
%     colormap(gray)
% end
%

k=0.1;
patch_size=400;
neurons=100;
batch_size=100;

W = randn(patch_size, neurons);
W = W*diag(1./sqrt(sum(W.^2,1)));

for j=1:200


    X=create_batch(I,patch_size,batch_size);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    b = W'*X;
    G = W'*W - eye(neurons);
    
    u = zeros(neurons,batch_size);
    
    for i =1:200
        
        a=u.*(abs(u) > k);
        
        u = 0.9 * u + 0.01 * (b - G*a);
             
    end
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   
    W = W + (5/size(X,2))*((X-W*a)*a');
    
    W = W*diag(1./sqrt(sum(W.^2,1)));
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    

end

    toc
    save('LCA_Dictionary.mat','W')



end


































function I=create_batch(Images,patch_size,batch_size)

[imsize, imsize, num_Images] = size(Images);

border=10;
patch_side = sqrt(patch_size);

I = zeros(patch_size,batch_size);

imi = ceil(num_Images * rand());

for i=1:batch_size
    
    row = border + ceil((imsize-patch_side-2*border) * rand());
    col = border + ceil((imsize-patch_side-2*border) * rand());
    
    I(:,i) = reshape(Images(row:row+patch_side-1, col:col+patch_side-1, imi),[patch_size, 1]);
    
    
end


end


























