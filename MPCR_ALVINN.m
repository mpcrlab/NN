
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
%
% Machine Perception and Cognitive Robotics Laboratory
%
%     Center for Complex Systems and Brain Sciences
%               Florida Atlantic University
%
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
%
% Vectorized Backpropagation Demonstration 
% Using ALVINN Data, See:
% Pomerleau, Dean A. Alvinn: An autonomous land vehicle in a neural network. 
% No. AIP-77. Carnegie-Mellon Univ Pittsburgh Pa 
% Artificial Intelligence And Psychology Project, 1989.
%
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function MPCR_ALVINN

clear all
close all
clc

load alvinn_data.mat


% for i=1:size(CVPatterns,2)
% 
% subplot(121)
% imagesc(reshape(CVPatterns(:,i),30,32))
% colormap(gray)
% subplot(122)
% plot(CVDesired(:,i),'bx')
% % plot(CVPositions(:,i),'bx')
% pause
% 
% end


pattern=CVPatterns';
category=CVDesired';

r=randperm(size(pattern,1));

pattern=pattern(r,:);
category=category(r,:);

category=(category-min(min(category)))/(max(max(category))-min(min(category)));

bias=ones(size(pattern,1),1);
pattern = [pattern bias];

n1 = size(pattern,2)
n2 = 6;   %n2-1
n3 = size(category,2);

w1 = 0.005*(1-2*rand(n1,n2-1));
w2 = 0.005*(1-2*rand(n2,n3));

dw1 = zeros(size(w1));
dw2 = zeros(size(w2));

L = 0.001;        % Learning
M = 0.5;          % Momentum

loop = 0;
sse=10;

figure(2)
surf(category)
view(0,80)
% imagesc(category)

while sse > 0.1
    
    act1 = [af(pattern * w1) bias];
    act2 = af(act1 * w2);
    
    error = category - act2;
    sse = sum(sum(error.^2))
    
%         plot(act2,'bx')
%         hold on
%         plot(category,'rx')
%         hold off


        figure(1)
        subplot(151)
        imagesc(reshape(w1(1:n1-1,1),30,32))
        subplot(152)
        imagesc(reshape(w1(1:n1-1,2),30,32))
        subplot(153)
        imagesc(reshape(w1(1:n1-1,3),30,32))
        subplot(154)
        imagesc(reshape(w1(1:n1-1,4),30,32))
        subplot(155)
        imagesc(reshape(w1(1:n1-1,5),30,32))
%     
%     
    figure(3)
    surf(act2)
    view(0,80)
    imagesc(act2)m


    drawnow()
 
    delta_w2 = error .* act2 .* (1-act2);
    delta_w1 = delta_w2*w2' .* act1 .* (1-act1);
    delta_w1(:,size(delta_w1,2)) = [];
    
  
    dw1 = L * pattern' * delta_w1 + M * dw1;
    dw2 = L * act1' * delta_w2 + M * dw2;
    
    
    w1 = w1 + dw1;
    w2 = w2 + dw2;
    
    
    loop = loop + 1;
    
    
end


end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------%
%--------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function action = af (weighted_sum)


action = 1./(1+exp(-weighted_sum));  		% Logistic / Sigmoid Function


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------%
%--------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



