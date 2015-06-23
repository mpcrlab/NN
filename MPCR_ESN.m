
function MPCR_ESN

clear all
close all
clc
tic

cd timeser
data = load('LORENZ.DAT');
% data = load('ROSSLER.DAT');
% data = load('HENON.DAT');
% data = load('EXPTPER.DAT');
% data = load('EXPTQP2.DAT');
% data = load('EXPTQP3.DAT');
% data = load('EXPTCHAO.DAT');

m = [floor(0.8*size(data,1)) floor(0.1*size(data,1)) floor(0.1*size(data,1))]; %time: training - transient - testing
n = [1 2000 1];  %nodes: network input - reservoir - output


a = 0.3;
r = 1e-8*eye(1+n(1)+n(2));

Wi = (rand(n(2),1+n(1))-0.5);
Wr = 0.1.*(rand(n(2),n(2))-0.5);

x = zeros(n(2),1);
X = zeros(1+n(1)+n(2),m(1)-m(2));
Y = zeros(n(3),m(3));

Yt = data(m(2)+2:m(1)+1)';

for t = 1:m(1)
   
	u = data(t);
    
	x = (1-a)*x + a*tanh( Wi*[1;u] + Wr*x );

	if t > m(2)
		X(:,t-m(2)) = [1;u;x];
    end
    
end

Wo=(Yt*X')/(X*X'+r);

u = data(m(1)+1);

for t = 1:m(3)
    
	x = (1-a)*x + a*tanh( Wi*[1;u] + Wr*x );
    
	y = Wo*[1;u;x];
    
	Y(:,t) = y;
	
    u = y;
    
end

toc

% error = 500;
% mse = sum((data(m(3)+2:m(3)+error+1)'-Y(1,1:error)).^2)./error


figure(1);
subplot(121)
plot(data(m(1)-4*m(3):m(1)-1));

subplot(122)
plot( data(m(1)+2:m(1)+m(3)+1), 'r');
hold on;
plot( Y', 'b' );
hold off;
axis tight;

end


