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
%Revised 4/1/13
%3D Chaotic Neuron
%********************************************************************%
function[]=MPCR_HR_Neuron()

clc
clear all
close all
tic

N=1;       %number of agents

t0=0;      %start time
tf=500;     %end time

numpts=10000;  %number of time steps even

h=(tf-t0)/(numpts-1); %define step size

t=t0:h:tf; %build time vector

% x=zeros(N,numpts); %allocate memory
% y=zeros(N,numpts);
% z=zeros(N,numpts);
%
% x=-1.7090+zeros(N,numpts); %allocate memory
% y=-13.5988+zeros(N,numpts);
% z=0.1543+zeros(N,numpts);

index=1;

x(:,index)=(rand(1,N)-rand(1,N))'; %initial conditions
y(:,index)=(rand(1,N)-rand(1,N))';
z(:,index)=(rand(1,N)-rand(1,N))';


% I=[zeros(1,numpts/2),zeros(1,numpts/2)];
I=[zeros(1,numpts/2),ones(1,numpts/2)];
% I=[zeros(1,numpts/4),2.*ones(1,numpts/4),2.*ones(1,numpts/4),2.*ones(1,numpts/4)];
% I=ones(1,numpts);

% plot(I)
%pause


%********************************************************************%
%Euler's method and Second Order Runge-Kutta
%********************************************************************%
for index=1:numpts-1
    
    
    x(:,index+1)=x(:,index)+h*dx_dt(x(:,index),y(:,index),z(:,index),I(index));
    y(:,index+1)=y(:,index)+h*dy_dt(x(:,index),y(:,index),z(:,index));
    z(:,index+1)=z(:,index)+h*dz_dt(x(:,index),y(:,index),z(:,index));
    
    index=index+1;
    
end

%********************************************************************%
%Plot
%********************************************************************%
%
%

plot(t,x,'LineWidth',1.25);


%  figure(1)
%  subplot(141)
%  plot(t,x,'LineWidth',1.25);
%
%
%  subplot(142)
%  plot(x,y,'LineWidth',1.25)
%
%  subplot(143)
%  plot(y,z,'LineWidth',1.25)
%
%  subplot(144)
%  plot(z,x,'LineWidth',1.25)

% scatter3(x,y,z)

toc

%********************************************************************%
%Functions to solve
%********************************************************************%
function[slope]=dx_dt(x,y,z,I)

%I -10 to 10

a=3;

slope=a*x.^2-x.^3+y-z+I;


function[slope]=dy_dt(x,y,z)

b=5;

slope=1-b*x.^2-y;


function[slope]=dz_dt(x,y,z)

r=0.001;
s=4;
xr=-8/5;


slope=r*(s*(x-xr)-z);







