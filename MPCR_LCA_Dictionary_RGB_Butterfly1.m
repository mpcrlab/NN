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
% Using RBG image data, see:
%
% 1)Rozell, Christopher J., et al.
% "Sparse coding via thresholding and
% local competition in neural circuits."
%
% 2)Whiten Images in Matlab
% http://xcorr.net/2013/04/30/whiten-images-in-matlab/
%
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function MPCR_LCA_Dictionary_RGB_Butterfly1

clear all
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ps=16;

for k=1:6
    
    foldernumber=k;
    
    switch foldernumber
        case 1
            foldername='admiral';
        case 2
            foldername='black_swallowtail';
        case 3
            foldername='machaon';
        case 4
            foldername='monarch_open';
        case 5
            foldername='peacock';
        case 6
            foldername='zebra';
        otherwise
            disp('error');
    end
    
    ps=16;
    ims=64;
    
    cd(['/Users/williamedwardhahn/Desktop/thesis/butterflies/' foldername '/'])
    
    ls
    
    dr1=dir('*.jpg')
    
    f1={dr1.name}; % get only filenames to cell
    X=[];
    
    for i=1:length(f1) % for each image
        
        i
        
        a1=f1{i};
        
        b1=imread(a1);
        
        imagesc(b1)
        
        pause
        
        s1=size(b1);
        
        if(s1(1) > s1(2))
            ns1=[ims NaN];
        else
            ns1=[NaN ims];
        end
        
        b1=im2double(imresize(b1,ns1));
        
        b1 = b1 - min(b1(:));
        b1 = b1 / max(b1(:));
        
        imagesc(b1)
        
        pause
        
        c1r=im2col(b1(:,:,1),[ps ps]);
        c1g=im2col(b1(:,:,2),[ps ps]);
        c1b=im2col(b1(:,:,3),[ps ps]);
        
        c=[c1r; c1b; c1g];
        
        X=[X, c];
        
    end
    
    X=X;
    X = bsxfun(@minus,X,mean(X)); %remove mean
    fX = fft(fft(X,[],2),[],3); %fourier transform of the images
    spectr = sqrt(mean(abs(fX).^2)); %Mean spectrum
    X = ifft(ifft(bsxfun(@times,fX,1./spectr),[],2),[],3); %whitened X
    
    save(['HahnColorPatches_' num2str(ps) '_Butterflies_' foldername '_whitened1.mat'],'X','-v7.3')
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

s=0.1;
patch_size=256*3;
neurons=256;
batch_size=100;
nk=6;
W = randn(patch_size, neurons, nk);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=1:6
      
    foldernumber=k;
    
    switch foldernumber
        case 1
            foldername='admiral';
        case 2
            foldername='black_swallowtail';
        case 3
            foldername='machaon';
        case 4
            foldername='monarch_open';
        case 5
            foldername='peacock';
        case 6
            foldername='zebra';
        otherwise
            disp('error');
    end
    
    
    cd('/Users/williamedwardhahn/Desktop/thesis/butterflies/butterflydata')
    
    data=load(['HahnColorPatches_' num2str(ps) '_Butterflies_' foldername '_whitened.mat'])
    
    X0=data.X;
    
    X0=sqrt(0.1)*X0/sqrt(mean(var(X0)));
    
    X1=X0(:,1:floor(end/2));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for j=1:500
        
        r=randperm(size(X1,2));
        
        X=X1(:,r(1:batch_size));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        W(:,:,k) = W(:,:,k)*diag(1./sqrt(sum(W(:,:,k).^2,1)));
        
        b = W(:,:,k)'*X;
        G = W(:,:,k)'*W(:,:,k) - eye(neurons);
        
        u = zeros(neurons,batch_size);
        
        for i =1:200
            
            a=u.*(abs(u) > s);
            
            u = 0.9 * u + 0.01 * (b - G*a);
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        W(:,:,k) = W(:,:,k) + (5/batch_size)*((X-W(:,:,k)*a)*a');
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        imagesc(filterplotcolor(W(:,:,k)')), drawnow()
        
        j
        
    end
    
end

save('Butterfly_LCA_W.mat','W')

load('Butterfly_LCA_W.mat')


s=0.15;
patch_size=256*3;
neurons=256;
batch_size=300;
nk=size(W,3);

for k=1:nk
    subplot(2,3,k)
    imagesc(filterplotcolor(W(:,:,k)')), drawnow()
end

figure(1)
drawnow()
pause

WW=[];

for k=1:size(W,3)
    
    WW=[WW W(:,:,k)];
    
end

% WW=randn(size(WW)); %Randomize weights to test

d=[];

for k=1:nk
    
    
    foldernumber=k;
    
    switch foldernumber
        case 1
            foldername='admiral';
        case 2
            foldername='black_swallowtail';
        case 3
            foldername='machaon';
        case 4
            foldername='monarch_open';
        case 5
            foldername='peacock';
        case 6
            foldername='zebra';
        otherwise
            disp('error');
    end
    
    
    cd('/Users/williamedwardhahn/Desktop/thesis/butterflies/butterflydata')
    
    data=load(['HahnColorPatches_' num2str(ps) '_Butterflies_' foldername '_whitened.mat']);
    
    X0=data.X;
    
    X0=sqrt(0.1)*X0/sqrt(mean(var(X0)));
    
    X2=X0(:,floor(end/2)+1:end);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
    WW = WW*diag(1./sqrt(sum(WW.^2,1)));
    G = WW'*WW - eye(neurons*nk);
    
    for j=1:20
        
        r=randperm(size(X2,2));
        
        X=X2(:,r(1:batch_size));
        
        for i=1:100
            imagesc(reshape(X(:,i),16,16,3))
            pause
        end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        b = WW'*X;
        
        u = zeros(neurons*nk,batch_size);
        
        for i =1:20
            
            a=u.*(abs(u) > s);
            
            u = 0.9 * u + 0.01 * (b - G*a);
            
        end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        a1=sum(a.^2,2);
        
        b=[];
        
        for j=1:nk
            
            b=[b sum(abs(a1(1+(j-1)*neurons:j*neurons)))];
            
        end
        
        subplot(121)
        bar(b)
        
        
        [b1,b2]=max(b);
        
        if sum(b)==0
            
            b2=-1;
            
        end
        
        d=[d b2==k];
        
        subplot(122)
        
        hist(d,0:1)
        
        drawnow()
        
    end
    
end

end





function [D] = filterplotcolor(W)

Dr=filterplot(W(:,1:size(W,2)/3));
Dg=filterplot(W(:,size(W,2)/3+1:2*size(W,2)/3));
Db=filterplot(W(:,2*size(W,2)/3+1:end));
D=zeros(size(Dr,1),size(Dr,2),3);
D(:,:,1)=Dr;
D(:,:,2)=Db;
D(:,:,3)=Dg;
D = D - min(D(:));
D = D / max(D(:));

end



function [D] = filterplot(X)

[m,n] = size(X);
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

