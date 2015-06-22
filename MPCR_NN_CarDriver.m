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
% William Hahn & Elan Barenholtz
% Feedback Amplifier 
% Backpropagation Neural Network
% Supervised Learning - Car and Driver Dataset
% August 26th, 2014
% Revised June 2, 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function MPCR_NN_CarDriver
clear all
close all
clc

load Hahn_CarDriver1.mat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Visualize Raw Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i=1:size(pattern,1)
%     pattern(i,:)
%     subplot(121)
%     imagesc(reshape(pattern(i,:),101,1175))
%     colormap(gray)
%     subplot(122)
%     imagesc(category(i,:))
%     pause
%     
% end

r=randperm(size(pattern,1)); %Shuffle Patterns

c=400; %Use the First 400 Patterns
pattern=pattern(r(1:c),:);
category=category(r(1:c),:);

bias=ones(size(pattern,1),1); %Add Bias (Default Resting State Potential)
pattern = [pattern bias];

n1 = size(pattern,2);   %Set the Number of Input Nodes Equal to Number of Pixels in the Input image
n2 = 10;   %n2-1        %Number of Hidden Nodes (Free Parameter)
n3 = size(category,2);  %Set the Number of Output Nodes Equal to the Number of Distinct Categories {left,forward,right}  

w1 = 0.005*(1-2*rand(n1,n2-1)); %Randomly Initialize Hidden Weights
w2 = 0.005*(1-2*rand(n2,n3));   %Randomly Initialize Output Weights

dw1 = zeros(size(w1));          %Set Initial Hidden Weight Changes to Zero
dw2 = zeros(size(w2));          %Set Initial Output Changes to Zero

L = 0.01;             % Learning    %Avoid Overshooting Minima 
M = 0.9;           % Momentum    %Smooths out the learning landscape

sse=size(pattern,1);  % Set Error Large so that Loop Starts
sseplot=[size(pattern,1) size(pattern,1) size(pattern,1)]; %Convergence Plot 

for loop=1:100
    
    loop
    
    b=1;
%   b=0.1;

    act1 = [af(b*pattern * w1) bias];     
    act2 = af(act1 * w2);

    error = category - act2;  %Calculate Error

    sse = sum(sum(error.^2)); % Error Reports - Not used by Algorithm 
    sseplot=[sseplot sse];
    
    delta_w2 = error .* act2 .* (1-act2); %Backpropagate Errors 
    delta_w1 = delta_w2*w2' .* act1 .* (1-act1);
    delta_w1(:,size(delta_w1,2)) = []; %Remove Bias
 
    dw1 = L * pattern' * delta_w1 + M * dw1; %Calculate Hidden Weight Changes 
    dw2 = L * act1' * delta_w2 + M * dw2;    %Calculate Output Weight Changes 

    w1 = w1 + dw1; %Adjust Hidden Weights 
    w2 = w2 + dw2; %Adjust Output Weights
    
%     w1 = w1 + 0.0005*(1-2*randn(n1,n2-1)); 
%     w2 = w2 + 0.0005*(1-2*randn(n2,n3)); 
%  
%     w1=w1/norm(w1);
%     w2=w2/norm(w2);
  
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Visualize Input Weights as Receptive Fields
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     if mod(loop,50)==0
%     figure(1)
%     set(gcf,'color','w');
    for i =1:n2-1
        subplot(sqrt(n2-1),sqrt(n2-1),i)
        imagesc(reshape(w1(1:n1-1,i),101,1175))
%         colormap(gray)
        axis off
    end
    
%     end
    
    
    
    
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Visualize Network Performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if mod(loop,5)==0

%     figure(2) 
%     set(gcf,'color','w');
%     
%     subplot(141)
%     imagesc(category)
%     
%     subplot(142)
%     imagesc(WTA(act2))
%     
%     subplot(143)
%     imagesc(abs(category-WTA(act2)))
%     
%     subplot(144)
%     plot(sseplot)
    
    drawnow()
    
% end

    
end %end for loop


end %end function




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------%
%--------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function action = af (weighted_sum)


action = 1./(1+exp(-weighted_sum));  		% Logistic / Sigmoid Function


end




function x = WTA(x)


for i=1:size(x,1)

[a,b]=max(x(i,:));

x(i,:)=1:size(x,2)==b;

end


end




