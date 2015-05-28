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
% Oscillator Neural Network for Image Segmentation and Binding
% 
% See:
% Yu, Guoshen, and J-J. Slotine. 
% Visual grouping by neural oscillator networks. 
% Neural Networks, IEEE Transactions on 20.12 (2009): 1871-1884.
%  
% Wang, DeLiang, and D. Termani. 
% Locally excitatory globally inhibitory oscillator networks. 
% Neural Networks, IEEE Transactions on 6.1 (1995): 283-286.
%
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%********************************************************************%
function[]=MPCR_Oscillator_Neural_Network()

clc
clear all
close all
%clf
tic

%********************************************************************%
% Synthetic Image for Oscillator Segmentation
%********************************************************************%
% Demo1
n=20;
A=zeros(n);
A(6:15,6:15)=255;
A(1:5,11:20)=127;
A(16:20,11:20)=127;
A(6:15,16:20)=127;

%********************************************************************%
% Demo2
% n=20;
% A=127.*ones(n);
% A(10,[4:8 13:17])=0;

%********************************************************************%
% Demo3
% s=40;
% n=s;
% c=100;
% d=5;
% 
% A=256*ones(s);
% 
% for j = [3 9]
% 
% for i = 1:6
% A((c*j+s*i):c*j+d+s*i)=0;
% end
% 
% end

%********************************************************************%
% Demo5
% n=2^5;
% A = phantom('Modified Shepp-Logan',n);

%********************************************************************%

u=A(1:end);

black=find(u==0);
gray=find(u==127);
white=find(u==255);

%Add Noise
%%
A=A+50.*(rand(size(A))-rand(size(A)));
u=A(1:end);
%%


figure(1)
imshow(A, [0 255],'InitialMagnification','fit');
% imshow(A,'InitialMagnification','fit');


umin=min(u);
umax=max(u);

Imin=0.8;
Imax=2;
beta=3;
omega=10;

I=(u-umin).*((Imax-Imin)/(umax-umin))+Imin;

figure(3)
% hist(u)
plot(u,'bx')


%********************************************************************%
%********************************************************************%
%Build Network Coupling Matrix
N=n^2; % Number of Oscillators in Network
% 
% xy=[1:N;1:N]';
% d = zeros(N);
% 
% for k = 1:N
%     
%     d(k,:) = sqrt(sum((xy(k*ones(N,1),:) - xy).^2,2)) < omega;
%     
% end
% 
% 
% uu=u(:,ones(1,size(u,1)));
% 
% 
% k=-10*ones(N);
% 
% k=k+exp((-(abs(uu'-uu)).^2)/(beta^2)).*d;

% k=k*0.00015;
% k=0.001*ones(N);
% k=rand(N)<0.0005;

k=zeros(N); %Uncoupled Network 
% figure(12)
% imshow(k)

%********************************************************************%
%********************************************************************%

t0=0;
tf=10;

numpts=tf*2000;

filename='oscillators';

h=(tf-t0)/(numpts-1); %define step size

t=t0:h:tf; %build time vector

v=zeros(N,numpts);
w=zeros(N,numpts);


index=1;

 v(:,index)=0.002*(rand(1,N)-rand(1,N))';
 w(:,index)=0.002*(rand(1,N)-rand(1,N))';


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

%********************************************************************%
%********************************************************************%

 if    mod(index,10)==0
% %     
    figure(2)
%     low=min(v(:,index));
%     high=max(v(:,index));
%     level=(high-low)/2;
%     imshow(reshape(w(:,index),n,n),'InitialMagnification','fit');
% 
%     
    
    low=min(v(:,index));
    high=max(v(:,index));
    level=(high-low)/2;
%   imshow(reshape(w(:,index),n,n),'InitialMagnification','fit');
    surf(reshape(w(:,index),n,n));
%     shading interp;
    zlim([-100 100])
    
end
    

%     colormap(jet)
%     pause(0.05);
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
% figure(6)
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
%
figure(6)
%plot(t,v([1,2,106,107,254,255],:),'LineWidth',1.25);
%plot(t,v([1,2,3,4,125,126,127,128,250,251,252,255],:),'LineWidth',1.25);
plot(t,v,'LineWidth',1.25);

figure(7)
% plot(v(:,end),w(:,end),'k.','MarkerSize',12);

hold on
plot(v(white,end),w(white,end),'r.','MarkerSize',12);
plot(v(gray,end),w(gray,end),'g.','MarkerSize',12);
plot(v(black,end),w(black,end),'b.','MarkerSize',12);
pause(0.1);
hold off

figure(8)

t=1:10:size(v,2);

hold on

for k=1:10

scatter3(v(white(k),t),w(white(k),t),t)
scatter3(v(gray(k),t),w(gray(k),t),t)
scatter3(v(black(k),t),w(black(k),t),t)

end

hold off

% figure(9)
% plot(v,'rx')

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

% slope=3.*v-v.^3-v.^7+2-w+I';

slope=3.*v-v.^3+2-w+I';

vv=v(:,ones(size(v,1),1));

slope=slope+sum(k.*(vv'-vv),2);

end

%********************************************************************%

function[slope]=dw_dt(v,w,k)

alpha=120;
c=0.4;
rho=4;
% 
% alpha=200;
% c=4;
% rho=4;

slope=c.*(alpha.*(tanh(rho.*v))-w);

ww=w(:,ones(size(w,1),1));

%slope=slope+sum(k.*(ww'-ww),2);

end


%********************************************************************%
%********************************************************************%

