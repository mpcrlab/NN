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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function MPCR_NN_2D_Landscape
clear all
close all
clc

[X,Y,Z] = Hahn_landscapes(3);

X = X - min(X(:));
X = X / max(X(:));

Y = Y - min(Y(:));
Y = Y / max(Y(:));

Z = Z - min(Z(:));
Z = Z / max(Z(:));

Z=0.5*Z;

figure(1)
surf(X,Y,Z,'EdgeColor','none')
view(-144,30)
pause

pattern=randi(size(Z,1),500,2);
testpattern=randi(size(Z,1),500,2);

category=[];
testcategory=[];

for i = 1:size(pattern,1)
    category=[category; Z(pattern(i,1),pattern(i,2))];  
end

for i = 1:size(testpattern,1) 
    testcategory=[testcategory; Z(testpattern(i,1),testpattern(i,2))];   
end

plot(pattern(:,1),pattern(:,2),'x')
scatter3(pattern(:,1),pattern(:,2),category)
pause

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Visualize Raw Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

r=randperm(size(pattern,1)); %Shuffle Patterns

c=500; %Use the First 400 Patterns
pattern=pattern(r(1:c),:);
category=category(r(1:c),:);

bias=ones(size(pattern,1),1); %Add Bias (Default Resting State Potential)
pattern = [pattern bias];
testpattern = [testpattern bias];

n1 = size(pattern,2);   %Set the Number of Input Nodes Equal to Number of Pixels in the Input image
n2 = 100;   %n2-1        %Number of Hidden Nodes (Free Parameter)
n3 = size(category,2);  %Set the Number of Output Nodes Equal to the Number of Distinct Categories {left,forward,right}

w1 = 0.005*(1-2*rand(n1,n2-1)); %Randomly Initialize Hidden Weights
w2 = 0.005*(1-2*rand(n2,n3));   %Randomly Initialize Output Weights

dw1 = zeros(size(w1));          %Set Initial Hidden Weight Changes to Zero
dw2 = zeros(size(w2));          %Set Initial Output Changes to Zero

L = 0.0001;         % Learning Rate    %Avoid Overshooting Minima
M = 0.9;            % Momentum         %Smooths out the learning landscape

sse=size(pattern,1);  % Set Error Large so that Loop Starts
sseplot=[size(pattern,1) size(pattern,1) size(pattern,1)]; %Convergence Plot

for loop=1:100000
    
    act1 = [af(pattern * w1) bias];
    act2 = af(act1 * w2);
    
    error = category - act2;  %Calculate Error
    
    delta_w2 = error .* act2 .* (1-act2); %Backpropagate Errors
    delta_w1 = delta_w2*w2' .* act1 .* (1-act1);
    delta_w1(:,size(delta_w1,2)) = []; %Remove Bias
    
    dw1 = L * pattern' * delta_w1 + M * dw1; %Calculate Hidden Weight Changes
    dw2 = L * act1' * delta_w2 + M * dw2;    %Calculate Output Weight Changes
    
    w1 = w1 + dw1; %Adjust Hidden Weights
    w2 = w2 + dw2; %Adjust Output Weights

    
    
    
    act22 = af([af(testpattern * w1) bias] * w2);
    
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------%
%Plots
%--------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(121)
scatter3(pattern(:,1),pattern(:,2),act2, 'bx')
view(loop/10,30)
hold on
scatter3(pattern(:,1),pattern(:,2),category, 'go')
view(loop/10,30)
hold off
title('Training')
subplot(122)
scatter3(testpattern(:,1),testpattern(:,2),act22, 'rx')
view(loop/10,30)
hold on
scatter3(testpattern(:,1),testpattern(:,2),testcategory, 'ko')
view(loop/10,30)
hold off
title('Testing')
sse = sum(sum(error.^2)) % Error Reports - Not used by Algorithm
sseplot=[sseplot sse];
% pause(0.05)

drawnow()

    
    
end %end for loom

% plot(sseplot)

end %end function




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------%
%--------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function action = af (weighted_sum)
% plot(1./(1+exp(-(-10:0.1:10))))

action = 1./(1+exp(-weighted_sum));  		% Logistic / Sigmoid Function


end




function x = WTA(x)


for i=1:size(x,1)
    
    [a,b]=max(x(i,:));
    
    x(i,:)=1:size(x,2)==b;
    
end


end















function [X,Y,Z] = Hahn_landscapes(i)


size_search_space=1;

resolution=20;

[X,Y] = meshgrid(-size_search_space:size_search_space/resolution:size_search_space, -size_search_space:size_search_space/resolution:size_search_space);



Z=fitness(i,X,Y);







end



function z = fitness(c,x1,x2)


switch c
    case 0
        z=-2*exp(-.01*(x1-5).^2 - .01*(x2-5).^2);
    case 1
        z=20*(1 - exp(-0.2*sqrt(0.5*(x1.^2 + x2.^2))))- exp(0.5*(cos(2*pi*x1) + cos(2*pi*x2))) + exp(1);
    case 2
        z=(1.5 - x1 + x1.*x2).^2 + (2.25 - x1 + x1.*x2.^2).^2 + (2.625 - x1 + x1.*x2.^3).^2;
    case 3
        z=sin(x1).*exp((1-cos(x2)).^2) + cos(x2).*exp((1-sin(x1)).^2) + (x1-x2).^2;
    case 4
        z=(x1 + 2*x2 - 7).^2 + (2*x1 + x2 - 5).^2;
    case 5
        z=100*(x2 - 0.01*x1.^2 + 1) + 0.01*(x1 + 10).^2;
    case 6
        z=100*x2.^2 + 0.01*abs(x1 + 10);
    case 7
        z=100*sqrt(abs(x2 - 0.01*x1.^2)) + 0.01*abs(x1 + 10);
    case 8
        z=-((cos(x1).*cos(x2).*exp(abs(1 - sqrt(x1.^2 + x2.^2)/pi))).^2)/30;
    case 9
        z=x1.^2 - 12*x1 + 11 + 10*cos(pi*x1/2) + 8*sin(5*pi*x1/2) - 1/sqrt(5)*exp(-((x2 - 0.5).^2)/2);
    case 10
        z=(abs(sin(x1).*sin(x2).*exp(abs(100 - sqrt(x1.^2 + x2.^2)/pi))) + 1).^(-0.1);
    case 11
        z=-0.0001*(abs(sin(x1).*sin(x2).*exp(abs(100 - sqrt(x1.^2 + x2.^2)/pi))) + 1).^(0.1);
    case 12
        z=-(abs(sin(x1).*sin(x2).*exp(abs(100 - sqrt(x1.^2 + x2.^2)/pi))) + 1).^(-0.1);
    case 13
        z=0.0001*(abs(sin(x1).*sin(x2).*exp(abs(100 - sqrt(x1.^2 + x2.^2)/pi))) + 1).^(0.1);
    case 14
        z=(100*(x2 - x1.^3).^2 + (1 - x1).^2);
    case 15
        z=-cos(x1).*cos(x2).*exp(-((x1-pi).^2 + (x2-pi).^2));
    case 16
        z=-(x2+47).*sin(sqrt(abs(x2+x1/2+47)))-x1.*sin(sqrt(abs(x1-(x2+47))));
    case 17
        z=(1  + (x1 + x2 + 1).^2.*(19 - 14*x1 +  3*x1.^2 - 14*x2 +  6*x1.*x2 +  3*x2.^2)).*(30 + (2*x1 - 3*x2).^2.*(18 - 32*x1 + 12*x1.^2 + 48*x2 - 36*x1.*x2 + 27*x2.^2));
    case 18
        z=(x1.^2 + x2.^2)/200 - cos(x1).*cos(x2/sqrt(2)) + 1;
    case 19
        z=100*((-10*atan2(x2, x1)/2/pi).^2 + (sqrt(x1.^2 + x2.^2) - 1).^2);
    case 20
        z=-abs(sin(x1).*cos(x2).*exp(abs(1 - sqrt(x1.^2 + x2.^2)/pi)));
    case 21
        z=100*(x2 - x1.^2).^2 + (1 - x1).^2;
    case 22
        z=sin(3*pi*x1).^2 + (x1-1).^2.*(1 + sin(3*pi*x2).^2) + (x2-1).^2.*(1 + sin(2*pi*x2).^2);
    case 23
        z=0.26*(x1.^2 + x2.^2) - 0.48*x1.*x2;
    case 24
        z= sin(x1 + x2) + (x1-x2).^2 - 1.5*x1 + 2.5*x2 + 1;
    case 25
        z = 0.5  + (sin(x1.^2 + x2.^2).^2 - 0.5) ./ (1+0.001*(x1.^2 + x2.^2)).^2;
    case 26
        z=0.5  + (sin(x1.^2 - x2.^2).^2 - 0.5) ./ (1+0.001*(x1.^2 + x2.^2)).^2;
    case 27
        z=0.5  + (sin(cos(abs(x1.^2 - x2.^2))).^2 - 0.5) ./ (1+0.001*(x1.^2 + x2.^2)).^2;
    case 28
        z= 0.5  + (cos(sin(abs(x1.^2 - x2.^2))).^2 - 0.5) ./ (1+0.001*(x1.^2 + x2.^2)).^2;
    case 29
        z=-exp(-(abs(cos(x1).*cos(x2).*exp(abs(1 - sqrt(x1.^2 + x2.^2)/pi)))).^(-1));
    case 30
        z= x1.^2 + x2.^2 - 10*cos(2*pi*x1) - 10*cos(2*pi*x2) + 20;
    case 31
        z=(100*(x2 - x1.^2).^2 + (1 - x1).^2);
    case 32
        z=-x1.*sin(sqrt(abs(x1))) -x2.*sin(sqrt(abs(x2)));
    case 33
        z=(4 - 2.1*x1.^2 + x1.^4/3).*x1.^2 + x1.*x2 + (4*x2.^2 - 4).*x2.^2;
    case 34
        z=-4*abs(sin(x1).*cos(x2).*exp(abs(cos((x1.^2 + x2.^2)/200))));
    case 35
        z=2*x1.^2 - 1.05*x1.^4 + x1.^6/6 + x1.*x2 + x2.^2;
    case 36
        z=(x1.^2 + x2.^2 - 2*x1).^2 + x1/4;
    otherwise
        disp('derp');
end






end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Gaussian Well %Global minimum: f(x)=-2, x(i)=5, i=1:2

%z=-2*exp(-.01*(x1-5).^2 - .01*(x2-5).^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ackley funcion

%z=20*(1 - exp(-0.2*sqrt(0.5*(x1.^2 + x2.^2))))- exp(0.5*(cos(2*pi*x1) + cos(2*pi*x2))) + exp(1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Beale funcion

%z=(1.5 - x1 + x1.*x2).^2 + (2.25 - x1 + x1.*x2.^2).^2 + (2.625 - x1 + x1.*x2.^3).^2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Bird function

%z=sin(x1).*exp((1-cos(x2)).^2) + cos(x2).*exp((1-sin(x1)).^2) + (x1-x2).^2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Booth function

%z=(x1 + 2*x2 - 7).^2 + (2*x1 + x2 - 5).^2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Bukin function #2

%z=100*(x2 - 0.01*x1.^2 + 1) + 0.01*(x1 + 10).^2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Bukin function #4


%z=100*x2.^2 + 0.01*abs(x1 + 10);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Bukin function #6

%z=100*sqrt(abs(x2 - 0.01*x1.^2)) + 0.01*abs(x1 + 10);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Carrom table function

%z=-((cos(x1).*cos(x2).*exp(abs(1 - sqrt(x1.^2 + x2.^2)/pi))).^2)/30;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Chichinad%ze function


%z=x1.^2 - 12*x1 + 11 + 10*cos(pi*x1/2) + 8*sin(5*pi*x1/2) - 1/sqrt(5)*exp(-((x2 - 0.5).^2)/2);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cross function

%z=(abs(sin(x1).*sin(x2).*exp(abs(100 - sqrt(x1.^2 + x2.^2)/pi))) + 1).^(-0.1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cross-in-tray function

%z=-0.0001*(abs(sin(x1).*sin(x2).*exp(abs(100 - sqrt(x1.^2 + x2.^2)/pi))) + 1).^(0.1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cross-leg table function

%z=-(abs(sin(x1).*sin(x2).*exp(abs(100 - sqrt(x1.^2 + x2.^2)/pi))) + 1).^(-0.1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Crowned cross function

%z=0.0001*(abs(sin(x1).*sin(x2).*exp(abs(100 - sqrt(x1.^2 + x2.^2)/pi))) + 1).^(0.1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extended cube function


%z=sum(  100*(x2 - x1.^3).^2 + (1 - x1).^2, 1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Easom function


%z=-cos(x1).*cos(x2).*exp(-((x1-pi).^2 + (x2-pi).^2));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generalized egg holder function


%z=-(x2+47).*sin(sqrt(abs(x2+x1/2+47)))-x1.*sin(sqrt(abs(x1-(x2+47))));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Goldstein-Price function


%z=(1  + (x1 + x2 + 1).^2.*(19 - 14*x1 +  3*x1.^2 - 14*x2 +  6*x1.*x2 +  3*x2.^2)).*(30 + (2*x1 - 3*x2).^2.*(18 - 32*x1 + 12*x1.^2 + 48*x2 - 36*x1.*x2 + 27*x2.^2));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Griewank funcion


%z=(x1.^2 + x2.^2)/200 - cos(x1).*cos(x2/sqrt(2)) + 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Helical valley function


%z=100*((-10*atan2(x2, x1)/2/pi).^2 + (sqrt(x1.^2 + x2.^2) - 1).^2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Holder table function


%z=-abs(sin(x1).*cos(x2).*exp(abs(1 - sqrt(x1.^2 + x2.^2)/pi)));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Leon funcion


%z=100*(x2 - x1.^2).^2 + (1 - x1).^2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Levi function, #13


%z=sin(3*pi*x1).^2 + (x1-1).^2.*(1 + sin(3*pi*x2).^2) + (x2-1).^2.*(1 + sin(2*pi*x2).^2);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Matyas function


%z=0.26*(x1.^2 + x2.^2) - 0.48*x1.*x2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% McCormick function


%z= sin(x1 + x2) + (x1-x2).^2 - 1.5*x1 + 2.5*x2 + 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Modified Schaffer function, #1


%compute pythagorean sum only once
%x12x22 = x1.^2 + x2.^2;

% output function value
%z = 0.5  + (sin(x12x22).^2 - 0.5) ./ (1+0.001*x12x22).^2;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Modified Schaffer function, #2


%z=0.5  + (sin(x1.^2 - x2.^2).^2 - 0.5) ./ (1+0.001*(x1.^2 + x2.^2)).^2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Modified Schaffer function, #3


%z=0.5  + (sin(cos(abs(x1.^2 - x2.^2))).^2 - 0.5) ./ (1+0.001*(x1.^2 + x2.^2)).^2;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Modified Schaffer function, #4


%z= 0.5  + (cos(sin(abs(x1.^2 - x2.^2))).^2 - 0.5) ./ (1+0.001*(x1.^2 + x2.^2)).^2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pen holder function


%z=-exp(-(abs(cos(x1).*cos(x2).*exp(abs(1 - sqrt(x1.^2 + x2.^2)/pi)))).^(-1));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Rastrigin function


%z= x1.^2 + x2.^2 - 10*cos(2*pi*x1) - 10*cos(2*pi*x2) + 20;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Extended Rosenbruck's Banana-function


%z=sum(100*(x2 - x1.^2).^2 + (1 - x1).^2, 1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Schweffel function

%z=-x1.*sin(sqrt(abs(x1))) -x2.*sin(sqrt(abs(x2)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Six hump camel back function

%z=(4 - 2.1*x1.^2 + x1.^4/3).*x1.^2 + x1.*x2 + (4*x2.^2 - 4).*x2.^2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Testtube holder function


%z=-4*abs(sin(x1).*cos(x2).*exp(abs(cos((x1.^2 + x2.^2)/200))));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Three hump camel back function



%z=2*x1.^2 - 1.05*x1.^4 + x1.^6/6 + x1.*x2 + x2.^2;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% %zettle function

%z=(x1.^2 + x2.^2 - 2*x1).^2 + x1/4;





















%z=(-x(:,1).*sin(sqrt(abs(x(:,1))))-x(:,2).*sin(sqrt(abs(x(:,2))))).*(abs(x(:,1))<500).*(abs(x(:,2))<500);





%z=sum(x'.^2);




%z=20+x(1,:).^2-10.*cos(2.*pi.*x(1,:))+x(2,:).^2-10.*cos(2.*pi.*x(2,:));

%z=-2*exp(-.01*(x(1)-5).^2 - .01*(x(2)-5).^2);

%z=20*(1 - exp(-0.2*sqrt(0.5*(x(1).^2 + x(2).^2))))- exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + exp(1);












