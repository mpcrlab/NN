
function MPCR_ELM

close all
clear all
clc



x=[-20:.5:20];
% % y=10.*x+0.5;
% % y=x.^2;
% % y=-abs(sin(x/8));
y=sin(.2.*x);
% y=x.^2+20.*rand(size(x));
% y=exp(-0.02.*(x-4).^2);


x=(x/norm(x))';
y=(y/norm(y))';


r=randperm(size(x,1));
x=x(r);
y=y(r);

x1=x(1:end/2);
y1=y(1:end/2);
x2=x(end/2+1:end);
y2=y(end/2+1:end);

h=10;

W1=randn(size(x1,2)+1,h);

y02=(tanh(([x2 ones(size(x2,1),1)]*W1))*(tanh(([x1 ones(size(x1,1),1)]*W1))\y1));




plot(x2,y02,'bx','Markersize', 10)

hold on

plot(x1,y1,'ro','Markersize', 15)

y2
y02

end















