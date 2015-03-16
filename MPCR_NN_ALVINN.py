########################################################
#------------------------------------------------------#
#
# Machine Perception and Cognitive Robotics Laboratory
#   Center for Complex Systems and Brain Sciences
#           Florida Atlantic University
#
#------------------------------------------------------#
########################################################
#------------------------------------------------------#
#
# Vectorized Backpropagation Demonstration 
# Using ALVINN Data, See:
# Pomerleau, Dean A. Alvinn: 
# An autonomous land vehicle in a neural network. 
# No. AIP-77. Carnegie-Mellon Univ Pittsburgh Pa 
# Artificial Intelligence And Psychology Project, 1989.
#
#------------------------------------------------------#
########################################################

import time
import numpy as np
import scipy.io
import matplotlib.pyplot as plot

########################################################
def af( x ):
   return [1/(1+np.exp(-x))]
########################################################
   

########################################################
def drawplots():
  
  for i in range(5):
    plot.subplot(5, 1, i+1)
    plot.imshow((w1[0:n1-1,i]).reshape(32,30).transpose())
    
  plot.draw()
  plot.pause(0.00001)
    
  return
########################################################
  
  
Data=scipy.io.loadmat('/Users/mpcr/Desktop/NN/alvinn_data.mat')
pattern=Data['CVPatterns']
pattern=pattern.transpose()
category=Data['CVDesired']
category=category.transpose()


r=np.random.permutation(pattern.shape[0])

pattern=pattern[r,:]
category=category[r,:]

category=(category-category.min())/(category.max()-category.min())

bias=np.ones((pattern.shape[0],1))

pattern=np.concatenate((pattern,bias), axis=1)

n1=pattern.shape[1]
n2=6
n3=category.shape[1]

w1=0.005*(1-np.random.random((n1,n2-1)))
w2=0.005*(1-np.random.random((n2,n3)))

dw1=np.zeros(w1.shape)
dw2=np.zeros(w2.shape)

L=0.01
M=0.5

loop=0
sse=10 

plot.axis([0, 1000, 0, 1])
plot.ion()
plot.show()


for loop in range(100):
  act1=np.concatenate((np.matrix(np.array(af(np.dot(0.1*pattern,w1)))),bias), axis=1)
  act2=np.matrix(np.array(af(act1*w2)))
  error = category - act2
  sse=np.power(error,2).sum()
  delta_w2=np.multiply(error,np.multiply(act2,(1-act2)))
  delta_w1=np.multiply(delta_w2*w2.transpose(),np.multiply(act1,(1-act1)))
  delta_w1=np.delete(delta_w1,-1,1)
  dw1=L*pattern.transpose()*delta_w1+np.multiply(M,dw1)
  dw2=L*act1.transpose()*delta_w2+np.multiply(M,dw2)
  w1=w1+dw1
  w2=w2+dw2
  drawplots()
  print(sse)
  
  
plot.close("all")

exit

