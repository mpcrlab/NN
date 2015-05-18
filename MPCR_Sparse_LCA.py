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
# Locally Competitive Algorithms Demonstration 
# Using natural images data, see:
# Rozell, Christopher J., et al. 
# "Sparse coding via thresholding and
# local competition in neural circuits." 
# Neural computation 20.10 (2008): 2526-2563.
#
#------------------------------------------------------#
########################################################
from matplotlib import cm
import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt

###############################################################################
def LCA_Network():


  Images = scipy.io.loadmat('./IMAGES.mat')
  Images = Images['IMAGES']


  k=0.1  
  patch_size=64
  neurons=64
  batch_size=100

  W=init_weights(patch_size, neurons)

  for i in range(1000):
      
    x=create_batch(Images,patch_size,batch_size)
    
    a = LCA(x, W, k)
      
    W = update_weights(x,W,a)
    
    drawplots(W)

  return W
###############################################################################  
  
  
###############################################################################
def LCA(x, W, k):

  (N, M) = np.shape(W)

  b = np.dot(W.T, x)
  G = np.dot(W.T, W) - np.eye(M)

  u = np.zeros((M,np.shape(x)[1]))

  
  for i in range(100):
    
    a = u
    a[np.abs(a) < k] = 0
       
    u = 0.9 * u + 0.1 * (b - np.dot(G, a))


  return a
###############################################################################


###############################################################################
def update_weights(I,W,a):
    
    W = W + (5.0/I.shape[1]) * (np.dot((I-np.dot(W, a)), a.T)) 
    
    W = np.dot(W, np.diag(1/np.sqrt(np.sum(W**2, axis = 0))))
    
    
    return W
###############################################################################  
  
  
  
###############################################################################
def init_weights(patch_size, neurons):
    
    W = np.random.randn(patch_size, neurons)
    W = np.dot(W, np.diag(1/np.sqrt(np.sum(W**2, axis = 0))))
    
    return W
###############################################################################  


###############################################################################
def create_batch(Images,patch_size,batch_size):
    
    (imsize, imsize, num_Images) = np.shape(Images)
    
    border=10  
    patch_side = np.sqrt(patch_size)
    
    I = np.zeros((patch_size,batch_size))

    imi = np.ceil(num_Images * random.uniform(0, 1))

    for i in range(batch_size):
        
      row = border + np.ceil((imsize-patch_side-2*border) * random.uniform(0, 1))
      col = border + np.ceil((imsize-patch_side-2*border) * random.uniform(0, 1))

      I[:,i] = np.reshape(Images[row:row+patch_side, col:col+patch_side, imi-1], patch_size, 1)
    

    return I
###############################################################################  
  
  
###############################################################################  
def drawplots(W):

    patch_size = np.sqrt(W.shape[0])
    plot_size = np.sqrt(W.shape[1])
    
    image = np.zeros((patch_size*plot_size+plot_size,patch_size*plot_size+plot_size))
    
    for i in range(plot_size.astype(int)):
        for j in range(plot_size.astype(int)):
            patch = np.reshape(W[:,i*plot_size+j],(patch_size,patch_size))
            image[i*patch_size+i:i*patch_size+patch_size+i,j*patch_size+j:j*patch_size+patch_size+j] = patch/np.max(np.abs(patch))
            
#    plt.imshow(np.reshape(W[:,1],(patch_size,patch_size)), cmap=cm.Greys_r, interpolation="none")
    plt.imshow(image, cmap=cm.jet, interpolation="none")
    
    plt.show()
###############################################################################  
  
###############################################################################  
###############################################################################  
###############################################################################  
  
LCA_Network()

###############################################################################
