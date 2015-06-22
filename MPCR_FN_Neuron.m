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
%William Hahn & Elan Barenholtz
%******************************************************%
%Numerical Ordinary Differential Equation Solver
%03/02/05
%Revised 2/7/13 
%FitzHugh-Nagumo Neuron Model
%******************************************************%
function[]=MPCR_FN_Neuron()

clc
clear all
clf

N=2;
n=1:N;

k=-0.05.*ones(N);
%k=rand(N);

t0=0;
tf=40;

numpts=1000;

h=(tf-t0)/(numpts-1); %define step size

t=t0:h:tf; %build time vector

v=zeros(N,numpts);
w=zeros(N,numpts);

index=1;

v(:,index)=2*(rand(1,N)-rand(1,N))';
w(:,index)=4*(rand(1,N)-rand(1,N))';

%********************************************************************%
%Euler's method and Second Order Runge-Kutta
%********************************************************************%
for index=1:numpts-1
    
    
  v(:,index+1)=v(:,index)+h*dv_dt(v(:,index),w(:,index),k);
  w(:,index+1)=w(:,index)+h*dw_dt(v(:,index),w(:,index),k);

  index=index+1 

%     k0v=h*dv_dt(v(:,index),w(:,index),k);
%     k1v=h*dv_dt(v(:,index)+h*dv_dt(v(:,index),w(:,index),k),w(:,index)+h*dw_dt(v(:,index),w(:,index),k),k);
%     
%     k0w=h*dw_dt(v(:,index),w(:,index),k);
%     k1w=h*dw_dt(v(:,index)+h*dv_dt(v(:,index),w(:,index),k),w(:,index)+h*dw_dt(v(:,index),w(:,index),k),k);
%     
%     v(:,index+1)=v(:,index)+(.5)*(k0v+k1v);
%     w(:,index+1)=w(:,index)+(.5)*(k0w+k1w);
%     
%     index=index+1;
    
    
end


%********************************************************************%
%Plot
%********************************************************************%
% 
% 
figure(1)
plot(t,v(1:2,:),'LineWidth',1.25);

figure(2)
hold on
plot(v(1,:),w(1,:),'r','LineWidth',1.25);
plot(v(2,:),w(2,:),'g','LineWidth',1.25);
% plot(v(3,:),w(3,:),'b','LineWidth',1.25);


%********************************************************************%
%Functions to solve
%********************************************************************%
function[slope]=dv_dt(v,w,k)

slope=3.*v-v.^3-v.^7+2-w;

slope=slope+sum(k.*(repmat(v,1,size(v,1))'-repmat(v,1,size(v,1))),2);


function[slope]=dw_dt(v,w,k)

alpha=12;
c=0.04;
rho=4;

slope=c.*(alpha.*(tanh(rho.*v))-w);

slope=slope+sum(k.*(repmat(w,1,size(w,1))'-repmat(w,1,size(w,1))),2);







