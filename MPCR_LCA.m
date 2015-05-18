
function MPCR_LCA()

clear all

load patches.mat

load dict512.mat

D=Wp';


for i = 4 : 4
    
    y=data(:,i);
    
    
    yy=reshape(y,16,16);
    
    

    
    a=LCA(y,D,0.01)
    
    
 
    
    
end


end





function [a, u] = LCA(y, D, lambda)


t=.01;
h=.0001;

d = h/t;
u = zeros(size(D,2),1);


for i=1:300
    
    
    a = ( u - sign(u).*(lambda) ) .* ( abs(u) > (lambda) );
    
    
    u =   u + d * ( D' * ( y - D*a ) - u - a  ) ;


end




end


































