function MPCR_Stochastic_Gradient()
 
 
 
[X,Y] = meshgrid(-5:0.1:5,-5:0.1:5);
 
Z=f(X,Y);
 
surf(X,Y,Z)
 
hold on
 
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 
x = [5 5]';
 
 
h = 0.1;
 
for i=1:100
    
      
    xn = x - h*df(x,randi([1 2]));
  

    plot([x(1) xn(1)],[x(2) xn(2)],'wo-')
    
    
    x = xn;
    
        
end    

 

end
 
 
 
function y=f(x1,x2) 
 
y=x1.^2 + x1.*x2 + 3*x2.^2;
 
end
 
 
function g = df(x,r)
 
%f=x1.^2 + x1.*x2 + 3*x2.^2;
 
grd = [2*x(1) + x(2), x(1) + 6*x(2)];
 
g = zeros(size(x,1),1);
 
g(r) = grd(r);
 
end
 