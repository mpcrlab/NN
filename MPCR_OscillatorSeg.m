%William Hahn
%Nonlinear Oscillator Network for Simple Image Segmentation
%
%********************************************************************%
function[]=MPCR_OscillatorSeg()

clc
clear all
close all
tic

%********************************************************************%
% Synthetic Image for Oscillator Segmentation
%********************************************************************%
n=20;
% A=zeros(n);
A(6:15,6:15)=255;
A(1:5,11:20)=127;
A(16:20,11:20)=127;
A(6:15,16:20)=127;

%  rgb = imread('coins3.jpg');
%  gray = rgb2gray(rgb);
%  imhist(gray);
%  bw = gray>40;
%  imshow(bw);
%  imwrite(bw,'bw.jpg');
%  [L,num] = bwlabel(bw);
%  %A = im2uint8(L/num);
%  A=im2double(L/num);
%  
%  A=imresize(A,[n,n]);
%  
%  imshow(A);

B=imread('rubin.gif','gif');
% % B=imread('triangle.jpg','jpg');
% % 
%  B=imresize(B,[n n]);
% B=rgb2gray(B);
 A=im2double(B);


u=A(1:end);

 

black=find(u==0);
gray=find(u==127);
white=find(u==255);

%%
A=A+5.*(randn(size(A))-randn(size(A)));
u=A(1:end);
%%


figure(1)
imshow(A, [0 255],'InitialMagnification','fit');

pause

% 
% figure(10)
% hist(u);

umin=min(u);
umax=max(u);

Imin=0.8;
Imax=2;
beta=10;%10;
omega=3;%3;

I=(u-umin).*((Imax-Imin)/(umax-umin))+Imin;




%********************************************************************%
%********************************************************************%

N=n^2; % Number of Oscillators in Network

xy=[1:N;1:N]';
d = zeros(N);

for k = 1:N
    
    d(k,:) = sqrt(sum((xy(k*ones(N,1),:) - xy).^2,2)) < omega;
    
end


uu=u(:,ones(1,size(u,1)));


k=-0.5*ones(N);

k=k+2.*exp((-(abs(uu'-uu)).^2)/(beta^2)).*d;

k=k*0.001;

%k=0.001*rand(N);

%k=0.001*ones(N);


% 
% figure(12)
% imshow(k)
% 





%********************************************************************%
%********************************************************************%

t0=0;
tf=2;

numpts=200*tf;

filename='oscillators';

h=(tf-t0)/(numpts-1); %define step size

t=t0:h:tf; %build time vector

v=zeros(N,numpts);
w=zeros(N,numpts);


index=1;

% v(:,index)=0.01.*(rand(1,N)-rand(1,N))';
% w(:,index)=0.01.*(rand(1,N)-rand(1,N))';


figure(2)
low=min(v(:,index));
high=max(v(:,index));
imshow(reshape(v(:,index),n,n), [low high],'InitialMagnification','fit');drawnow;



%********************************************************************%
%Euler's method and Second Order Runge-Kutta
%********************************************************************%
%********************************************************************%

for index=1:numpts-1
    
    
      v(:,index+1)=v(:,index)+h*dv_dt(v(:,index),w(:,index),k,I);
      w(:,index+1)=w(:,index)+h*dw_dt(v(:,index),w(:,index),k);
    
      index=index+1;
    
%********************************************************************%    
%********************************************************************%    
%     k0v=h*dv_dt(v(:,index),w(:,index),k,I);
%     k1v=h*dv_dt(v(:,index)+h*dv_dt(v(:,index),w(:,index),k,I),w(:,index)+h*dw_dt(v(:,index),w(:,index),k),k,I);
%     
%     k0w=h*dw_dt(v(:,index),w(:,index),k);
%     k1w=h*dw_dt(v(:,index)+h*dv_dt(v(:,index),w(:,index),k,I),w(:,index)+h*dw_dt(v(:,index),w(:,index),k),k);
%     
%     v(:,index+1)=v(:,index)+(.5)*(k0v+k1v);
%     w(:,index+1)=w(:,index)+(.5)*(k0w+k1w);
%     
%     index=index+1;
%********************************************************************%
%********************************************************************%

    
% %     
    figure(2)
    %subplot(121)
%     low=min(v(:,index));
%     high=max(v(:,index));
%     level=(high-low)/2;
%       imshow(reshape(v(:,index),n,n),[low high],'InitialMagnification','fit');
surf(reshape(v(:,index),n,n));      
% surf(reshape(v(:,index),n,n), 'LineStyle', 'none');
%     shading interp;
    zlim([-10 10])
%     axis('square', 'off');
%     grid('off');
%       colormap(jet)

    %pause(0.001);
%     
%     figure(5)
%     subplot(411)
%     plot(t,v(white,:),'r','LineWidth',1.25);
%     subplot(412)
%     plot(t,v(gray,:),'g','LineWidth',1.25);
%     subplot(413)
%     plot(t,v(black,:),'b','LineWidth',1.25);
%     subplot(414)
%     hold on
%     plot(t,v(white,:),'r','LineWidth',0.25);
%     plot(t,v(gray,:),'g','LineWidth',0.25);
%     plot(t,v(black,:),'b','LineWidth',0.25);
%     hold off
%     drawnow;
% 
% 
% 
% subplot(122)
% hold on
% plot(v(white,index),w(white,index),'r.','MarkerSize',12);
% plot(v(gray,index),w(gray,index),'g.','MarkerSize',12);
% plot(v(black,index),w(black,index),'b.','MarkerSize',12);
% pause(0.1);
% hold off

% figure(7)
% plot(v(:,index),w(:,index),'k.','MarkerSize',12);
% pause(0.05);


end


%********************************************************************%
%Plot
%********************************************************************%
%



surf(v([black,gray,white],:))
colormap(jet)
shading interp;



%
%figure(3)
%plot(t,v([1,2,106,107,254,255],:),'LineWidth',1.25);
%plot(t,v([1,2,3,4,125,126,127,128,250,251,252,255],:),'LineWidth',1.25);
%plot(t,v,'LineWidth',1.25);


% 
% figure(4)
% hold on
% plot(v(white,:),w(white,:),'r','LineWidth',1.25);
% plot(v(gray,:),w(gray,:),'g','LineWidth',1.25);
% plot(v(black,:),w(black,:),'b','LineWidth',1.25);
% hold off
% 
% 
figure(5)
subplot(511)
plot(t,v(white,:),'r','LineWidth',1.25);
subplot(512)
plot(t,v(gray,:),'g','LineWidth',1.25);
subplot(513)
plot(t,v(black,:),'b','LineWidth',1.25);
subplot(514)
hold on
plot(t,v(white,:),'r','LineWidth',0.25);
plot(t,v(gray,:),'g','LineWidth',0.25);
plot(t,v(black,:),'b','LineWidth',0.25);
hold off
subplot(515)
thresh=1.29;
hold on
plot(t,v(white,:)>thresh,'r','LineWidth',0.25);
plot(t,v(gray,:)>thresh,'g','LineWidth',0.25);
plot(t,v(black,:)>thresh,'b','LineWidth',0.25);
hold off

% 
% 
% 
% 
% 
% 
% figure(6)
% hold on
% plot(v(white,end),w(white,end),'r.','MarkerSize',12);
% plot(v(gray,end),w(gray,end),'g.','MarkerSize',12);
% plot(v(black,end),w(black,end),'b.','MarkerSize',12);
% 
% 
% 
% figure(7)
% plot(v(:,end),w(:,end),'k.','MarkerSize',12);




%
% X=[v(:,end),w(:,end)];
%
% figure(5)
% plot(X)
%
% [label,centers]=kmeans(X,3);
%
% hold on
% plot(X(label==1,1),X(label==1,2),'r.','MarkerSize',12)
% plot(X(label==2,1),X(label==2,2),'b.','MarkerSize',12)
% plot(X(label==3,1),X(label==3,2),'g.','MarkerSize',12)



toc

end


%********************************************************************%
%Functions to solve
%********************************************************************%
function[slope]=dv_dt(v,w,k,I)

slope=3.*v-v.^3-v.^7+2-w+I';

vv=v(:,ones(size(v,1),1));

slope=slope+sum(k.*(vv'-vv),2);

end

%********************************************************************%

function[slope]=dw_dt(v,w,k)

% alpha=12;
% c=0.4;
% rho=4;

% alpha=200;
% c=4;
% rho=4;

alpha=210.04;
c=0.3964;
rho=22.72;


slope=c.*(alpha.*(-1+2./(1+exp(-2*(rho.*v))))-w);

%slope=c.*(alpha.*(tanh(rho.*v))-w);

ww=w(:,ones(size(w,1),1));

slope=slope+sum(k.*(ww'-ww),2);

end






%********************************************************************%
%********************************************************************%

