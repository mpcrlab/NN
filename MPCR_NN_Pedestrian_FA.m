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
% Pedestrian Detection
% Vectorized Backpropagation Neural Network Demonstration 
% See:
% Daimler Pedestrian Classification Benchmark Dataset
% S. Munder and D. M. Gavrila. 
% An Experimental Study on Pedestrian Classification. 
% IEEE Transactions on Pattern Analysis and Machine Intelligence, 
% vol. 28, no. 11, pp.1863-1868, November 2006.
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
function MPCR_NN_Pedestrian_FA

clear all
clc

load Daimler_ped_data.mat

pattern=[im2double(c1) im2double(c2)]';

category=[ones(1,size(c1,2)) 0.*ones(1,size(c2,2))]';

r=randperm(size(pattern,1));

pattern=pattern(r,:);
category=category(r);

r=randperm(size(pattern,1))

pattern=pattern(r,:);
category=category(r);

trainpattern=pattern(1:end/3,:);

testpattern=pattern(end/3+1:2*(end/3),:);

validpattern=pattern(2*(end/3)+1:end,:);

traincategory=category(1:end/3,:);

testcategory=category(end/3+1:2*(end/3),:);

validcategory=category(2*(end/3)+1:end,:);


% 
% for i =1:size(pattern,1)
% 
% imagesc(reshape(pattern(i,:),60,64))
% colormap(gray)
% 
% category(i)
% 
% pause
% 
% end
% 
% 
% return

bias=ones(size(trainpattern,1),1);
trainpattern = [trainpattern bias];

testpattern = [testpattern bias];
validpattern = [validpattern bias];


n1 = size(trainpattern,2)
n2 = 2;   %n2-1
n3 = size(traincategory,2);

w1 = 0.001*(1-2*rand(n1,n2-1));
w2 = 0.001*(1-2*rand(n2,n3));

dw1 = zeros(size(w1));
dw2 = zeros(size(w2));

L = 0.0001;        % Learning
M = 0.8;          % Momentum




p=0.01;

errorplot=[];

for loop=1:1000
    
%   b=1-p*abs(randn());
%   b=0.25
    b=1;
    
    act1 = [af((b*trainpattern) * w1) bias];
    act2 =  af((act1) * w2);
    
    error = traincategory - act2;
    sse = sum(error.^2); 
    
    delta_w2 = error .* act2 .* (1-act2);
    delta_w1 = delta_w2*w2' .* act1 .* (1-act1);
    delta_w1(:,size(delta_w1,2)) = [];
    
    
    dw1 = L * trainpattern' * delta_w1 + M * dw1;
    dw2 = L * act1' * delta_w2 + M * dw2;
    
    w1 = w1 + dw1;
    w2 = w2 + dw2;
    

    
    
    
    
    
    
    figure(1)
%     surf(reshape(w1(1:n1-1,1),36,18))
    subplot(121)
    imagesc(reshape(w1(1:n1-1,1),36,18))
%     colormap(jet)
    subplot(122)
    %     imagesc(reshape(w1(1:n1-1,2),36,18))
    %     subplot(163)
    %     plot(w1)
    subplot(174)
    
    nn=1:length(testcategory);
    plot(nn(find(traincategory==1)),act2(find(traincategory==1)),'g^','MarkerSize',12)
    hold on
    plot(nn(find(traincategory==0)),act2(find(traincategory==0)),'rv','MarkerSize',12)
    
    plot(traincategory,'bo')
    hold off
    %
    
    
    subplot(175)
    act1 = [af(testpattern * w1) bias];
    %
    act2 = af(act1 * w2);
    
    %
    nn=1:length(testcategory);
    plot(nn(find(testcategory==1)),act2(find(testcategory==1)),'g^','MarkerSize',12)
    hold on
    plot(nn(find(testcategory==0)),act2(find(testcategory==0)),'rv','MarkerSize',12)
    
    plot(testcategory,'bo')
    hold off
    
    error2 = testcategory ~= (act2>0.5);
    sse2 = sum(error2);
    
    
    subplot(176)
    act1 = [af(validpattern * w1) bias];
    %
    act2 = af(act1 * w2);
   
    %
    nn=1:length(testcategory);
    plot(nn(find(validcategory==1)),act2(find(validcategory==1)),'g^','MarkerSize',12)
    hold on
    plot(nn(find(validcategory==0)),act2(find(validcategory==0)),'rv','MarkerSize',12)
    
    hold on
    plot(validcategory,'bo')
    hold off
    
    error3 = validcategory ~= (act2>0.5);
    sse3 = sum(error3);
    
    ep1=[loop b (1-(sse/length(testcategory)))*100 (1-(sse2/length(testcategory)))*100 (1-(sse3/length(validcategory)))*100]
    errorplot=[errorplot; ep1];
    subplot(177)
    plot(errorplot(:,[3,4,5]))

    set(gcf,'color','w');

    drawnow()

 
end


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------%
%--------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function network_output = neural_network(neural_input,hidden_weights,output_weights)

network_output=af([af(([neural_input,1])*hidden_weights),1]*output_weights);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------%
%--------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





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



