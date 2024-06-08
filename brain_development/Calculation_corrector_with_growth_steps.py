import pickle
import os
import jax.numpy as jnp
from jax import random as rd
key = rd.PRNGKey(0)
import random
import jax.nn as nn
import numpy as np
import time
import jax
from jax import grad, jit, vmap,jacfwd,remat
from functools import partial
import optax
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
####
#######initialize parameters
def init_params_b_shrinkage(layers, key,k):
  Ws = []
  bs = []
  for i in range(len(layers) - 1):
    std_glorot = jnp.sqrt(2/(layers[i] + layers[i + 1]))
    key, subkey = jax.random.split(key)
    Ws.append(jax.random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    bs.append(jnp.zeros(layers[i + 1]))
  shrinkage=k*jnp.ones((1,))
  return (Ws, bs,shrinkage)
@jit
def forward_pass_b(H, params):
  Ws = params[0]
  bs = params[1]
  N_layers = len(Ws)
  for i in range(N_layers - 1):
    H = jnp.matmul(H, Ws[i]) + bs[i]
    H = jnp.tanh(H)
  Y = jnp.matmul(H, Ws[-1]) + bs[-1]
  return Y
def image_boundary (x,x_min:int,x_max:int):
    id_boundary=jnp.where((x_min<x)&(x<x_max) ,1,0)
    id_boundary1=jnp.round(x* id_boundary)
    #x=torch.where(id_boundary==0,torch.ones_like(x),x)
    #x
    return id_boundary,id_boundary1
@jit
def s1(X1X2X3,img):
    X1=X1X2X3[:,[0]]*co_tr_S1[0,0]+co_tr_S1[0,1]
    X2=X1X2X3[:, [1]] * co_tr_S1[1, 0] + co_tr_S1[1, 1]
    X3=X1X2X3[:, [2]] * co_tr_S1[2, 0] + co_tr_S1[2, 1]
    x1=X1
    #x2=x2_max-X2
    x2 =X2
    x3=X3
    x1_floor = jnp.floor(x1).astype(int)
    x1_ceil = x1_floor + 1
    i01, i1 = image_boundary(x1_floor, x1_min, x1_max_S1)
    i02, i2= image_boundary(x1_ceil, x1_min, x1_max_S1)
    x2_floor = jnp.floor(x2).astype(int)
    x2_ceil = x2_floor + 1
    j01, j1 = image_boundary(x2_floor, x2_min, x2_max_S1)
    j02, j2= image_boundary(x2_ceil, x2_min, x2_max_S1)
    x3_floor = jnp.floor(x3).astype(int)
    x3_ceil = x3_floor + 1
    k01, k1 = image_boundary(x3_floor, x3_min, x3_max_S1)
    k02, k2=  image_boundary(x3_ceil, x3_min, x3_max_S1)
    ans1=k02* j02* i01*img[k2, j2, i1] * (x1 - x1_ceil) * (x2 - x2_floor) *(x3-x3_floor)/ (
            (x1_floor - x1_ceil) * (x2_ceil- x2_floor)*(x3_ceil-x3_floor))
    ans2 =k02* j01* i01*img[ k2, j1, i1] * (x1 - x1_ceil) * (x2 - x2_ceil) * (
                x3 - x3_floor) / (
                   (x1_floor - x1_ceil) * (x2_floor - x2_ceil) * (x3_ceil - x3_floor))
    ans3=k02* j01*i02*img[ k2, j1, i2] * (x1 - x1_floor) * (x2 - x2_ceil) * (
                x3 - x3_floor) / (
                   (x1_ceil-x1_floor ) * (x2_floor - x2_ceil) * (x3_ceil - x3_floor))
    ans4=k02*j02* i02*img[ k2, j2, i2] * (x1 - x1_floor) * (x2 - x2_floor) * (
                x3 - x3_floor) / (
                   (x1_ceil-x1_floor ) * (x2_ceil-x2_floor ) * (x3_ceil - x3_floor))
    ans5 =k01*j01*i02*img[k1, j1, i2] * (x1 - x1_floor) * (x2 - x2_ceil) * (
            x3 - x3_ceil) / (
                   (x1_ceil - x1_floor) * (x2_floor-x2_ceil) * (x3_floor-x3_ceil))
    ans6 = k01* j02* i02*img[k1, j2, i2] * (x1 - x1_floor) * (x2 - x2_floor) * (
            x3 - x3_ceil) / (
                   (x1_ceil - x1_floor) * (x2_ceil - x2_floor) * (x3_floor - x3_ceil))
    ans7 = k01* j02* i01*img[k1, j2, i1] * (x1 - x1_ceil) * (x2 - x2_floor) * (
            x3 - x3_ceil) / (
                   (x1_floor-x1_ceil) * (x2_ceil - x2_floor) * (x3_floor - x3_ceil))
    ans8 = k01* j01* i01*img[k1, j1, i1] * (x1 - x1_ceil) * (x2 - x2_ceil) * (
            x3 - x3_ceil) / (
                   (x1_floor - x1_ceil) * (x2_floor-x2_ceil) * (x3_floor - x3_ceil))
    return ans1+ans2+ans3+ans4+ans5+ans6+ans7+ans8
@jit
def s2(x1x2x3,img):
    x1 =x1x2x3[:, [0]] * co_tr_S2[0, 0] + co_tr_S2[0, 1]
    x2 = x1x2x3[:, [1]] * co_tr_S2[1, 0] + co_tr_S2[1, 1]
    x3 = x1x2x3[:, [2]] * co_tr_S2[2, 0] + co_tr_S2[2, 1]
    x1 = x1
    #x2 =x2_max- X2
    x2 =x2
    x3=x3
    x1_floor = jnp.floor(x1).astype(int)
    x1_ceil = x1_floor + 1
    i01, i1 = image_boundary(x1_floor, x1_min, x1_max_S2)
    i02, i2= image_boundary(x1_ceil, x1_min, x1_max_S2)
    x2_floor = jnp.floor(x2).astype(int)
    x2_ceil = x2_floor + 1
    j01, j1 = image_boundary(x2_floor, x2_min, x2_max_S2)
    j02, j2= image_boundary(x2_ceil, x2_min, x2_max_S2)
    x3_floor = jnp.floor(x3).astype(int)
    x3_ceil = x3_floor + 1
    k01, k1 = image_boundary(x3_floor, x3_min, x3_max_S2)
    k02, k2=  image_boundary(x3_ceil, x3_min, x3_max_S2)
    ans1=k02* j02* i01*img[k2, j2, i1] * (x1 - x1_ceil) * (x2 - x2_floor) *(x3-x3_floor)/ (
            (x1_floor - x1_ceil) * (x2_ceil- x2_floor)*(x3_ceil-x3_floor))
    ans2 =k02* j01* i01*img[ k2, j1, i1] * (x1 - x1_ceil) * (x2 - x2_ceil) * (
                x3 - x3_floor) / (
                   (x1_floor - x1_ceil) * (x2_floor - x2_ceil) * (x3_ceil - x3_floor))
    ans3=k02* j01*i02*img[ k2, j1, i2] * (x1 - x1_floor) * (x2 - x2_ceil) * (
                x3 - x3_floor) / (
                   (x1_ceil-x1_floor ) * (x2_floor - x2_ceil) * (x3_ceil - x3_floor))
    ans4=k02*j02* i02*img[ k2, j2, i2] * (x1 - x1_floor) * (x2 - x2_floor) * (
                x3 - x3_floor) / (
                   (x1_ceil-x1_floor ) * (x2_ceil-x2_floor ) * (x3_ceil - x3_floor))
    ans5 =k01*j01*i02*img[k1, j1, i2] * (x1 - x1_floor) * (x2 - x2_ceil) * (
            x3 - x3_ceil) / (
                   (x1_ceil - x1_floor) * (x2_floor-x2_ceil) * (x3_floor-x3_ceil))
    ans6 = k01* j02* i02*img[k1, j2, i2] * (x1 - x1_floor) * (x2 - x2_floor) * (
            x3 - x3_ceil) / (
                   (x1_ceil - x1_floor) * (x2_ceil - x2_floor) * (x3_floor - x3_ceil))
    ans7 = k01* j02* i01*img[k1, j2, i1] * (x1 - x1_ceil) * (x2 - x2_floor) * (
            x3 - x3_ceil) / (
                   (x1_floor-x1_ceil) * (x2_ceil - x2_floor) * (x3_floor - x3_ceil))
    ans8 = k01* j01* i01*img[k1, j1, i1] * (x1 - x1_ceil) * (x2 - x2_ceil) * (
            x3 - x3_ceil) / (
                   (x1_floor - x1_ceil) * (x2_floor-x2_ceil) * (x3_floor - x3_ceil))
    return ans1+ans2+ans3+ans4+ans5+ans6+ans7+ans8
@jit
def dNN_dxs(x1x2x3,nn_f):
    dNN1_dxs_= vmap(grad(lambda x1x2x3_: forward_pass_b(x1x2x3_, nn_f)[0]))
    dNN2_dxs_= vmap(grad(lambda x1x2x3_: forward_pass_b(x1x2x3_, nn_f)[1]))
    dNN3_dxs_= vmap(grad(lambda x1x2x3_: forward_pass_b(x1x2x3_, nn_f)[2]))
    return jnp.hstack((dNN1_dxs_(x1x2x3),dNN2_dxs_(x1x2x3),dNN3_dxs_(x1x2x3)))
@jit
def mesh_address(x,x_min,x_max,n):
    params=jnp.array([-n*x_min/(x_max-x_min),n/(x_max-x_min)])
    map=params[0]+params[1]*x
    return [jnp.floor(map),jnp.floor(map)+1]
def ODE_Euler_end_time(X1X2X3,nn_f):
    x1x2x3=X1X2X3
    for i in range(time_steps_pre):
        x1x2x3_change=forward_pass_b(x1x2x3,nn_f)
        x1x2x3=x1x2x3+ sudo_dt_pre * x1x2x3_change
    return x1x2x3
@jit
def D_function(x1x2x3,image_boundary_co):
    #x1x2=jnp.where(( x1x2[0]>x1_max) | (x1x2[0]<x1_min)|(x1x2[1]>x2_max) | (x1x2[1]<x2_min), jnp.zeros_like(x1x2),x1x2)
    coo_x1 = image_boundary_co[:, [0]] - x1x2x3[0]
    coo_x2 = image_boundary_co[:, [1]] - x1x2x3[1]
    coo_x3 = image_boundary_co[:, [2]] - x1x2x3[2]
    return jnp.min(jnp.sqrt(coo_x1 ** 2 + coo_x2 ** 2+coo_x3**2))
def Dgrad_calculations(image_boundary):
    x1_min=-2.5*155/189
    x1_max=2.5*155/189
    x2_min=-2.5
    x2_max=2.5
    x3_min=-2.5*135/189
    x3_max=2.5*135/189
    n_mesh_x=250
    n_mesh_y=250
    n_mesh_z=250
    x1 =np.linspace(x1_min,x1_max,n_mesh_x)
    x2 =np.linspace(x2_min,x2_max,n_mesh_y)
    x3 =np.linspace(x3_min,x3_max,n_mesh_z)
    x2,x3,x1=np.meshgrid(x2,x3,x1,indexing='xy')
    x1_shape=x1.shape
    x1 = jnp.reshape(x1, (x1.size, 1))
    x2= jnp.reshape(x2, (x2.size, 1))
    x3= jnp.reshape(x3, (x3.size, 1))
    x1x2x3=jnp.hstack((x1,x2,x3))
    DGrad_function_vmap=vmap(jacfwd(lambda  x1x2X3_:D_function( x1x2X3_,image_boundary)))
    ans=np.zeros((1,3))
    for i in range(1000):
        x1x2x3_var=x1x2x3[125*125*i:125*125*i+125*125,:]
        #print(i)
        ans=np.append(ans,np.array(DGrad_function_vmap(x1x2x3_var)),axis=0)
    ans=ans[1:,:]
    # ans2=D_function_vmap(x1x2x3)
    Grad_x1=jnp.reshape(ans[:,[0]],x1_shape)
    Grad_x2=jnp.reshape(ans[:,[1]],x1_shape)
    Grad_x3=jnp.reshape(ans[:,[2]],x1_shape)
    D_Grad_data=[Grad_x1,Grad_x2,Grad_x3]
    print(np.array(Grad_x1).shape)
    print(np.array(Grad_x2).shape)
    print(np.array(Grad_x3).shape)
    print(np.array(Grad_x1).max())
    print(np.array(Grad_x2).max())
    print(np.array(Grad_x3).max())
    return D_Grad_data
@jit
def DGrad_interpolation(x1x2x3,x1_min,x1_max,x2_min,x2_max,x3_min,x3_max,n,data):
    data_x1=data[0]
    data_x2=data[1]
    data_x3=data[2]
    x1x2x3=jnp.where(( x1x2x3[:,[0]]>x1_max) | (x1x2x3[:,[0]]<x1_min)|(x1x2x3[:,[1]]>x2_max) | (x1x2x3[:,[1]]<x2_min)|
                     (x1x2x3[:,[2]]>x3_max) | (x1x2x3[:,[2]]<x3_min), jnp.zeros_like(x1x2x3),x1x2x3)
    x1=x1x2x3[:,[0]]
    x2=x1x2x3[:,[1]]
    x3=x1x2x3[:,[2]]
    nx1_floor, nx1_ceil = mesh_address(x1, x1_min, x1_max, n)
    nx1_floor = nx1_floor.astype(int)
    nx1_ceil = nx1_ceil.astype(int)
    x1_floor = x1_min + nx1_floor * (x1_max - x1_min) / (n - 1)
    x1_ceil = x1_min + nx1_ceil * (x1_max - x1_min) / (n - 1)
    nx2_floor, nx2_ceil = mesh_address(x2, x2_min, x2_max, n)
    nx2_floor = nx2_floor.astype(int)
    nx2_ceil = nx2_ceil.astype(int)
    x2_floor = x2_min + nx2_floor * (x2_max - x2_min) / (n - 1)
    x2_ceil = x2_min + nx2_ceil * (x2_max - x2_min) / (n - 1)
    nx3_floor, nx3_ceil = mesh_address(x3, x3_min, x3_max, n)
    nx3_floor = nx3_floor.astype(int)
    nx3_ceil = nx3_ceil.astype(int)
    x3_floor = x3_min + nx3_floor * (x3_max - x3_min) / (n - 1)
    x3_ceil = x3_min + nx3_ceil * (x3_max - x3_min) / (n - 1)
    ans1= data_x1[nx3_floor,nx2_floor, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_ceil)*(x3 - x3_ceil) / (
            (x1_floor - x1_ceil) * (x2_floor - x2_ceil)*(x3_floor - x3_ceil)) + \
          data_x1[nx3_floor,nx2_floor, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_ceil)*(x3 - x3_ceil) / (
                  (x1_ceil - x1_floor) * (x2_floor - x2_ceil)*(x3_floor - x3_ceil)) + \
          data_x1[nx3_floor,nx2_ceil, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_floor)*(x3 - x3_ceil) / (
                  (x1_ceil - x1_floor) * (x2_ceil - x2_floor)*(x3_floor - x3_ceil)) + \
          data_x1[nx3_floor,nx2_ceil, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_floor) *(x3 - x3_ceil)/ (
                  (x1_floor - x1_ceil) * (x2_ceil - x2_floor)*(x3_floor - x3_ceil))+\
          data_x1[nx3_ceil,nx2_floor, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_ceil) *(x3 - x3_floor)/ (
            (x1_floor - x1_ceil) * (x2_floor - x2_ceil)*(x3_ceil-x3_floor)) + \
          data_x1[nx3_ceil,nx2_floor, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_ceil) *(x3 - x3_floor)/ (
                  (x1_ceil - x1_floor) * (x2_floor - x2_ceil)*(x3_ceil-x3_floor)) + \
          data_x1[nx3_ceil,nx2_ceil, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_floor) *(x3 - x3_floor)/ (
                  (x1_ceil - x1_floor) * (x2_ceil - x2_floor)*(x3_ceil-x3_floor)) + \
          data_x1[nx3_ceil,nx2_ceil, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_floor)*(x3 - x3_floor) / (
                  (x1_floor - x1_ceil) * (x2_ceil - x2_floor)*(x3_ceil-x3_floor))

    ans2 = data_x2[nx3_floor,nx2_floor, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_ceil)*(x3 - x3_ceil) / (
            (x1_floor - x1_ceil) * (x2_floor - x2_ceil)*(x3_floor - x3_ceil)) + \
          data_x2[nx3_floor,nx2_floor, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_ceil)*(x3 - x3_ceil) / (
                  (x1_ceil - x1_floor) * (x2_floor - x2_ceil)*(x3_floor - x3_ceil)) + \
          data_x2[nx3_floor,nx2_ceil, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_floor)*(x3 - x3_ceil) / (
                  (x1_ceil - x1_floor) * (x2_ceil - x2_floor)*(x3_floor - x3_ceil)) + \
          data_x2[nx3_floor,nx2_ceil, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_floor) *(x3 - x3_ceil)/ (
                  (x1_floor - x1_ceil) * (x2_ceil - x2_floor)*(x3_floor - x3_ceil))+\
          data_x2[nx3_ceil,nx2_floor, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_ceil) *(x3 - x3_floor)/ (
            (x1_floor - x1_ceil) * (x2_floor - x2_ceil)*(x3_ceil-x3_floor)) + \
          data_x2[nx3_ceil,nx2_floor, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_ceil) *(x3 - x3_floor)/ (
                  (x1_ceil - x1_floor) * (x2_floor - x2_ceil)*(x3_ceil-x3_floor)) + \
          data_x2[nx3_ceil,nx2_ceil, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_floor) *(x3 - x3_floor)/ (
                  (x1_ceil - x1_floor) * (x2_ceil - x2_floor)*(x3_ceil-x3_floor)) + \
          data_x2[nx3_ceil,nx2_ceil, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_floor)*(x3 - x3_floor) / (
                  (x1_floor - x1_ceil) * (x2_ceil - x2_floor)*(x3_ceil-x3_floor))
    
    ans3 = data_x3[nx3_floor,nx2_floor, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_ceil)*(x3 - x3_ceil) / (
            (x1_floor - x1_ceil) * (x2_floor - x2_ceil)*(x3_floor - x3_ceil)) + \
          data_x3[nx3_floor,nx2_floor, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_ceil)*(x3 - x3_ceil) / (
                  (x1_ceil - x1_floor) * (x2_floor - x2_ceil)*(x3_floor - x3_ceil)) + \
          data_x3[nx3_floor,nx2_ceil, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_floor)*(x3 - x3_ceil) / (
                  (x1_ceil - x1_floor) * (x2_ceil - x2_floor)*(x3_floor - x3_ceil)) + \
          data_x3[nx3_floor,nx2_ceil, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_floor) *(x3 - x3_ceil)/ (
                  (x1_floor - x1_ceil) * (x2_ceil - x2_floor)*(x3_floor - x3_ceil))+\
          data_x3[nx3_ceil,nx2_floor, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_ceil) *(x3 - x3_floor)/ (
            (x1_floor - x1_ceil) * (x2_floor - x2_ceil)*(x3_ceil-x3_floor)) + \
          data_x3[nx3_ceil,nx2_floor, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_ceil) *(x3 - x3_floor)/ (
                  (x1_ceil - x1_floor) * (x2_floor - x2_ceil)*(x3_ceil-x3_floor)) + \
          data_x3[nx3_ceil,nx2_ceil, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_floor) *(x3 - x3_floor)/ (
                  (x1_ceil - x1_floor) * (x2_ceil - x2_floor)*(x3_ceil-x3_floor)) + \
          data_x3[nx3_ceil,nx2_ceil, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_floor)*(x3 - x3_floor) / (
                  (x1_floor - x1_ceil) * (x2_ceil - x2_floor)*(x3_ceil-x3_floor))
    return ans1,ans2,ans3
@remat
@jit
def xs_and_Fs(x1x2x3_p_mini,Fp_values_mini,XB_i,D_grad_data_i,nn_f):
    Fs_c=Fp_values_mini
    eye=jnp.array([1,0,0,0,1,0,0,0,1])
    x1x2x3=x1x2x3_p_mini
    for i in range(time_steps):
        D_function_vmap=vmap(lambda x1x2x3_:D_function(x1x2x3_,XB_i))
        D_values_mini = D_function_vmap(x1x2x3)[:,None]
        # x1x2x3_for_grad=jnp.where(D_values_mini>.00001,x1x2x3,x1x2x3+.00000007*x1x2x3)
        # DG=DGrad_function_vmap(x1x2x3_for_grad)
        D_Gradx1_values_mini, D_Gradx2_values_mini,D_Gradx3_values_mini =\
             DGrad_interpolation(x1x2x3, x1_min_D, x1_max_D, x2_min_D,  x2_max_D, x3_min_D,x3_max_D,n_i, D_grad_data_i)
        # D_Gradx1_values_mini=DG[:,[0]]; D_Gradx2_values_mini=DG[:,[1]];D_Gradx3_values_mini=DG[:,[2]]
        g1g2g3_values = forward_pass_b(x1x2x3, nn_f)
        dNN_dxs_values = dNN_dxs(x1x2x3, nn_f)
        g_times_dD_dX = jnp.hstack((g1g2g3_values[:, [0]] * D_Gradx1_values_mini, g1g2g3_values[:, [0]] * D_Gradx2_values_mini,g1g2g3_values[:, [0]] * D_Gradx3_values_mini,\
                                    g1g2g3_values[:, [1]] * D_Gradx1_values_mini, g1g2g3_values[:, [1]] * D_Gradx2_values_mini,g1g2g3_values[:, [1]] * D_Gradx3_values_mini,\
                                    g1g2g3_values[:, [2]] * D_Gradx1_values_mini, g1g2g3_values[:, [2]] * D_Gradx2_values_mini,g1g2g3_values[:, [2]] * D_Gradx3_values_mini))
        Fs_c_n=eye+ sudo_dt * g_times_dD_dX  +sudo_dt *D_values_mini *dNN_dxs_values
        Fs_c=jnp.hstack((Fs_c_n[:,[0]]*Fs_c[:,[0]]+Fs_c_n[:,[1]]*Fs_c[:,[3]]+Fs_c_n[:,[2]]*Fs_c[:,[6]],\
                         Fs_c_n[:,[0]]*Fs_c[:,[1]]+Fs_c_n[:,[1]]*Fs_c[:,[4]]+Fs_c_n[:,[2]]*Fs_c[:,[7]],\
                         Fs_c_n[:,[0]]*Fs_c[:,[2]]+Fs_c_n[:,[1]]*Fs_c[:,[5]]+Fs_c_n[:,[2]]*Fs_c[:,[8]],\
                         Fs_c_n[:,[3]]*Fs_c[:,[0]]+Fs_c_n[:,[4]]*Fs_c[:,[3]]+Fs_c_n[:,[5]]*Fs_c[:,[6]],\
                         Fs_c_n[:,[3]]*Fs_c[:,[1]]+Fs_c_n[:,[4]]*Fs_c[:,[4]]+Fs_c_n[:,[5]]*Fs_c[:,[7]],\
                         Fs_c_n[:,[3]]*Fs_c[:,[2]]+Fs_c_n[:,[4]]*Fs_c[:,[5]]+Fs_c_n[:,[5]]*Fs_c[:,[8]],\
                         Fs_c_n[:,[6]]*Fs_c[:,[0]]+Fs_c_n[:,[7]]*Fs_c[:,[3]]+Fs_c_n[:,[8]]*Fs_c[:,[6]],\
                         Fs_c_n[:,[6]]*Fs_c[:,[1]]+Fs_c_n[:,[7]]*Fs_c[:,[4]]+Fs_c_n[:,[8]]*Fs_c[:,[7]],\
                         Fs_c_n[:,[6]]*Fs_c[:,[2]]+Fs_c_n[:,[7]]*Fs_c[:,[5]]+Fs_c_n[:,[8]]*Fs_c[:,[8]]))
        x1x2x3=x1x2x3+ sudo_dt * D_values_mini*forward_pass_b(x1x2x3,nn_f)
    return x1x2x3,Fs_c
def x_value(X1X2X3,nn_f):
    x1x2x3=X1X2X3
    for i in range(time_steps_pre):
        x1x2x3=x1x2x3+ sudo_dt_pre * forward_pass_b(x1x2x3,nn_f)
    return x1x2x3
def F_values(X1X2X3,nn_f):
    dx1dX=vmap(grad(lambda x1x2x3: x_value(x1x2x3, nn_f)[0]))
    dx2dX = vmap(grad(lambda x1x2x3:x_value(x1x2x3, nn_f)[1]))
    dx3dX = vmap(grad(lambda x1x2x3:x_value(x1x2x3, nn_f)[2]))
    F11_12_13=dx1dX(X1X2X3)
    F21_22_23=dx2dX(X1X2X3)
    F31_32_33=dx3dX(X1X2X3)
    return jnp.hstack((F11_12_13[:,[0]],F11_12_13[:,[1]],F11_12_13[:,[2]],F21_22_23[:,[0]],F21_22_23[:,[1]],F21_22_23[:,[2]],
                       F31_32_33[:,[0]],F31_32_33[:,[1]],F31_32_33[:,[2]]))
def predictor_and_gradiants(X1X2X3, F0,nn_f_p):
    F_values_p=F_values(X1X2X3, nn_f_p)
    F_values_p_material= jnp.hstack((F_values_p[:,[0]] * F0[:, [0]] + F_values_p[:,[1]]  * F0[:, [3]]+F_values_p[:,[2]]  * F0[:, [6]],\
                                     F_values_p[:,[0]] * F0[:, [1]] + F_values_p[:,[1]]  * F0[:, [4]]+F_values_p[:,[2]]  * F0[:, [7]],\
                                     F_values_p[:,[0]] * F0[:, [2]] + F_values_p[:,[1]]  * F0[:, [5]]+F_values_p[:,[2]]  * F0[:, [8]],\
                                     F_values_p[:,[3]] * F0[:, [0]] + F_values_p[:,[4]]  * F0[:, [3]]+F_values_p[:,[5]]  * F0[:, [6]],\
                                     F_values_p[:,[3]] * F0[:, [1]] + F_values_p[:,[4]]  * F0[:, [4]]+F_values_p[:,[5]]  * F0[:, [7]],\
                                     F_values_p[:,[3]] * F0[:, [2]] + F_values_p[:,[4]]  * F0[:, [5]]+F_values_p[:,[5]]  * F0[:, [8]],\
                                     F_values_p[:,[6]] * F0[:, [0]] + F_values_p[:,[7]]  * F0[:, [3]]+F_values_p[:,[8]]  * F0[:, [6]],\
                                     F_values_p[:,[6]] * F0[:, [1]] + F_values_p[:,[7]]  * F0[:, [4]]+F_values_p[:,[8]]  * F0[:, [7]],\
                                     F_values_p[:,[6]] * F0[:, [2]] + F_values_p[:,[7]]  * F0[:, [5]]+F_values_p[:,[8]]  * F0[:, [8]]))
    x1x2x3_p=ODE_Euler_end_time(X1X2X3,nn_f_p)
    return [ x1x2x3_p, F_values_p_material]
@jit
def psi( Fs,S1_binary_gauss_mini,sf):
    Fs=Fs*jnp.exp(-jnp.exp(sf))
    F11 = Fs[:,[0]]
    F12 = Fs[:,[1]]
    F13 = Fs[:,[2]]
    F21 = Fs[:,[3]]
    F22 = Fs[:,[4]]
    F23 = Fs[:,[5]]
    F31 = Fs[:,[6]]
    F32 = Fs[:,[7]]
    F33 = Fs[:,[8]]
    C11 = F11 * F11 + F21 * F21+F31 * F31
    C12 = F11*F12 + F21*F22 + F31*F32
    C13=F11*F13 + F21*F23 + F31*F33
    C21 = C12
    C22 = F12 * F12 + F22 * F22+F32*F32
    C23=F12*F13 + F22*F23 + F32*F33
    C31=C13
    C32=C23
    C33= F13*F13 + F23*F23 + F33*F33
    trace_C = C11 + C22+C33
    trace_C=jnp.where(S1_binary_gauss_mini==1, trace_C, 3*jnp.ones_like(trace_C))
    det_F = F11*F22*F33 - F11*F23*F32 - F12*F21*F33 + F12*F23*F31 + F13*F21*F32 - F13*F22*F31
    det_F=jnp.where(S1_binary_gauss_mini==1, det_F, jnp.ones_like(det_F))
    energy = (.5 * muu * (trace_C- 3) - muu * jnp.log(det_F) + .5 * lam * (jnp.log(det_F)) ** 2)
    return energy
@jit
def s1_s2(S1_binary_gauss_mini,X1X2X3_mini,x1x2x3,S1_image,S2_image):
    s1_X = s1(X1X2X3_mini,S1_image)
    s1_X = jnp.where(S1_binary_gauss_mini==1, s1_X, jnp.zeros_like(s1_X))
    s2_phi_X = s2(x1x2x3,S2_image)
    s2_phi_X = jnp.where(S1_binary_gauss_mini == 1, s2_phi_X, jnp.zeros_like(s2_phi_X))
    ans = (s1_X - s2_phi_X) ** 2
    return ans
@jit
def integral(nn_f,X1X2X3_mini, x1x2x3_p_mini,Fp_values_mini,XB_i,D_grad_data_i,S1_image,S2_image,S1_binary_gauss_mini):
    x1x2x3,Fs=xs_and_Fs(x1x2x3_p_mini,Fp_values_mini,XB_i,D_grad_data_i,(nn_f[0],nn_f[1]))
    coef=jnp.ones_like(S1_binary_gauss_mini) * dx * dy *dz/ 8
    var0=s1_s2(S1_binary_gauss_mini,X1X2X3_mini,x1x2x3,S1_image,S2_image)
    var1=psi(Fs,S1_binary_gauss_mini,nn_f[2])
    second_integrand=S1_binary_gauss_mini* var1#S1_gauss
    first_integral=jnp.trace(jnp.matmul(jnp.transpose(var0), coef))
    second_integral=jnp.trace(jnp.matmul(jnp.transpose(second_integrand), coef))
    return [second_integral, first_integral, second_integral,[],[]]
@jit
def loss_computation(nn_f,X1X2X3_mini, x1x2x3_p_mini,Fp_values_mini,XB_i,D_grad_data_i,S1_image,S2_image,S1_binary):
    vrbl = integral(nn_f,X1X2X3_mini, x1x2x3_p_mini,Fp_values_mini,XB_i,D_grad_data_i,S1_image,S2_image,S1_binary)
    loss0 = vrbl[0]
    return [loss0, vrbl[1], vrbl[2], vrbl[3], vrbl[4]]
def training(nn_f,X1,X2,X3,x1x2x3_p,Fp_values,batch_size_x,idn,XB_i,D_grad_data_i,S1_image,S2_image,S1_binary,lr, epoch_max:int=1000):
    id_non=0
    loss_evolution = []
    loss_evolution_mismatch = []
    loss_evolution_energy = []
    loss_evolution_growth = []
    # loss_evolution =np.squeeze(np.load('Total_Loss_corrector_fig2.npy',allow_pickle=True)).tolist()
    # loss_evolution_mismatch =np.squeeze(np.load('Mismatch_Loss_corrector_fig2.npy',allow_pickle=True)).tolist()
    # loss_evolution_energy = np.squeeze(np.load('Energy_Loss_corrector_fig2.npy',allow_pickle=True)).tolist()
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(nn_f)
    start = time.time()
    for i in range(epoch_max+1):
        intrand = random.sample(idn, batch_size_x)
        intrand.sort()
        intrand = np.array(intrand)
        intrand = \
            np.sort(jnp.concatenate(( intrand,  intrand + 1), axis=0))
        intrand=jnp.transpose(intrand[None,:])
        X1_mini = jnp.squeeze(X1[intrand, :])
        X2_mini = jnp.squeeze(X2[intrand, :])
        X3_mini = jnp.squeeze(X3[intrand, :])
        # X1_O_mini=jnp.squeeze(X1_O[intrand, :])
        # X2_O_mini=jnp.squeeze(X2_O[intrand, :])
        # X3_O_mini=jnp.squeeze(X3_O[intrand, :])
        S1_binary_mini=jnp.squeeze(S1_binary[intrand, :])
        size = X1_mini.size
        X1_mini = jnp.reshape(X1_mini, (size, 1))
        X2_mini = jnp.reshape(X2_mini, (size, 1))
        X3_mini = jnp.reshape(X3_mini, (size, 1))
        S1_binary_mini=jnp.reshape(S1_binary_mini,(size, 1))
        X1X2X3_mini=jnp.hstack((X1_mini,X2_mini,X3_mini))
        # X1X2X3_O_mini=jnp.hstack((X1_O_mini.reshape((size, 1)),X2_O_mini.reshape((size, 1)),X3_O_mini.reshape((size, 1))))
        x1x2x3_p_mini= jnp.hstack((jnp.squeeze(x1x2x3_p[0][intrand, :]).reshape((size, 1)), jnp.squeeze(x1x2x3_p[1][intrand, :]).reshape((size, 1)),
                                  jnp.squeeze(x1x2x3_p[2][intrand, :]).reshape((size, 1))))
        Fp_values_mini= jnp.hstack((jnp.squeeze(Fp_values[0][intrand, :]).reshape((size, 1)), jnp.squeeze(Fp_values[1][intrand, :]).reshape((size, 1)), \
                                     jnp.squeeze(Fp_values[2][intrand, :]).reshape((size, 1)), jnp.squeeze(Fp_values[3][intrand, :]).reshape((size, 1)),
                                     jnp.squeeze(Fp_values[4][intrand, :]).reshape((size, 1)),jnp.squeeze(Fp_values[5][intrand, :]).reshape((size, 1)),
                                     jnp.squeeze(Fp_values[6][intrand, :]).reshape((size, 1)),jnp.squeeze(Fp_values[7][intrand, :]).reshape((size, 1)),
                                     jnp.squeeze(Fp_values[8][intrand, :]).reshape((size, 1))))
        try:
           a=loss_computation(nn_f,X1X2X3_mini,x1x2x3_p_mini,Fp_values_mini,XB_i,D_grad_data_i,S1_image,S2_image,S1_binary_mini)
           loss=a[0]
           if jnp.isnan(loss):
               id_non=id_non+1
           #print(a[0])
           if ~(jnp.isnan(loss) | jnp.isinf(loss)):
               grad_for_opt=lambda nn_f: loss_computation(nn_f,X1X2X3_mini,x1x2x3_p_mini,Fp_values_mini,XB_i,D_grad_data_i,S1_image,S2_image,S1_binary_mini)[0]
               grads = jax.grad(grad_for_opt)(nn_f)
               updates, opt_state = optimizer.update(grads, opt_state)
               nn_f = optax.apply_updates(nn_f, updates)
           if i % 10==0:
                print(f"id_non={id_non}")
                print(f'epoch={i}, loss={loss}')
                print(f"imag_mismatch={a[1]}")
                print(f"energy={a[2]}")
                print(f"det={a[3]}")
                print(f"log_det={a[4]}")
                print(f"growth={nn_f[2]}")
                loss_evolution.append(loss)
                loss_evolution_mismatch .append(a[1])
                loss_evolution_energy.append(a[2])
                loss_evolution_growth.append(nn_f[2])
        except KeyboardInterrupt:
            break
    return [nn_f,loss_evolution,loss_evolution_mismatch,loss_evolution_energy,loss_evolution_growth,id_non]
def weights_coordinates(x,y):
    n_mesh_x=int(x.shape[0])
    n_mesh_y=int(y.shape[0])
    x_bar = x[0:2]
    y_bar = y[0:2]
    for i in range(2, n_mesh_x):
        #x_bar=np.append(x_bar, x[i-1:i+1],axis=0)
        x_bar = np.vstack((x_bar, x[i - 1:i + 1]))
        #x_bar = torch.cat((x_bar, x[i-1:i+1]), dim=0)
    for i in range(2, n_mesh_y):
        #y_bar = np.append(y_bar, y[i - 1:i + 1], axis=0)
        y_bar = np.vstack((y_bar, y[i - 1:i + 1]))
        #y_bar = torch.cat((y_bar, y[i-1:i+1]), dim=0)
    X1,X2 =np.meshgrid(x_bar,y_bar,indexing='xy')
    return X1,X2
def gausspoints (X1,X2):
    X1G=(X1[:, [0]] * (1 + 1 / np.sqrt(3)) + X1[:, [0 + 1]] * (1 - 1 / np.sqrt(3))) / 2
    X2G=(X2[[0], :] * (1 + 1 / np.sqrt(3)) + X2[[0 + 1],:] * (1 - 1 / np.sqrt(3))) / 2
    for i in range(1,X1.shape[0]):
        if i % 2 == 0:
            X1G =np.hstack((X1G,(X1[:, [i]] * (1 + 1 / jnp.sqrt(3)) + X1[:, [i + 1]] * (1 - 1 / jnp.sqrt(3))) / 2))
            X2G = np.vstack((X2G, (X2[[i], :] * (1 + 1 / jnp.sqrt(3)) + X2[[i + 1], :] * (1 - 1 / jnp.sqrt(3))) / 2))
        else:
            X1G = np.hstack((X1G,(X1[:, [i - 1]] * (1 - 1 / jnp.sqrt(3)) + X1[:, [i]] * (1 + 1 / jnp.sqrt(3))) / 2))
            X2G = np.vstack((X2G, (X2[[i - 1],:] * (1 - 1 / jnp.sqrt(3)) + X2[[i],:] * (1 + 1 / jnp.sqrt(3))) / 2))
    return(X1G,X2G)
def column_coef(X1,X2,n_mesh_x,n_mesh_y):
    X1p=np.zeros((2,2))
    X2p = np.zeros((2,2))
    for j in range(n_mesh_y-1):
          print(j)
          for ii in range(n_mesh_x-1):
             X1p=np.vstack((X1p,X1[2*j:2*(j+1),2*ii:2*(ii+1)]))
             X2p=np.vstack((X2p, X2[ 2 * j:2 * (j + 1),2 * ii:2 * (ii + 1)]))
    X1p=X1p[2:,:]
    X2p = X2p[2:, :]
    return(X1p,X2p)
def column_coef_3D(X11,X22,X31,n_mesh_x,n_mesh_y,n_mesh_z):
    X1=X11.repeat(int(2*(n_mesh_z-1)),axis=0)
    X2 =X22.repeat(int(2*(n_mesh_z-1)),axis=0)
    X33_box=np.zeros((2*(n_mesh_z-1),2,2))
    X3=np.zeros_like(X1)
    id=0
    id2=0
    for i in range(n_mesh_z-1):
        
        X31_FG=(X31[id] * (1 +1/ np.sqrt(3))+X31[id + 1] * (1 -1/ np.sqrt(3)))/2
        X31_SG = (X31[id] * (1 - 1 / np.sqrt(3)) + X31[id + 1] * (1 + 1 / np.sqrt(3)))/2
        vrbl = np.array([[[X31_FG, X31_FG], [X31_FG, X31_FG]], [[X31_SG, X31_SG], [X31_SG, X31_SG]]])
        X33_box[id2:id2+2,:,:]=vrbl 
        id+=1
        id2+=2
    id=0
    for i in range((n_mesh_x-1)*(n_mesh_y-1)):
        X3[id:id+2*(n_mesh_z-1),:,:]=X33_box
        id+=2*(n_mesh_z-1)
    return(X1,X2,X3)
with open('NN_t_pres_fig8.pickle', 'rb') as f:
     NN_t_pres = pickle.load(f)
with open('BP.pickle', 'rb') as f:
     XB= pickle.load(f)
n_i=250
### program starts from here
time_steps=15
sudo_dt=1/time_steps
time_steps_pre=15
sudo_dt_pre=1/time_steps_pre
with open('image_S1_0_binary.pickle', 'rb') as f:
     S1_binary = pickle.load(f)
# with open('image_S1_0.pickle', 'rb') as f:
#      S1_0 = pickle.load(f)
# with open('image_S1_1.pickle', 'rb') as f:
#      S1_1 = pickle.load(f)
# with open('image_S1_2.pickle', 'rb') as f:
#      S1_2 = pickle.load(f)
# with open('image_S1_3.pickle', 'rb') as f:
#      S1_3 = pickle.load(f)
# with open('image_S1_4.pickle', 'rb') as f:
#      S1_4 = pickle.load(f)
# with open('image_S1_5.pickle', 'rb') as f:
#      S1_5= pickle.load(f)
# S1_0=S1_0/S1_0.max()
# S1_1=S1_1/S1_1.max()
# S1_2=S1_2/S1_2.max()
# S1_3=S1_3/S1_3.max()
# S1_4=S1_4/S1_4.max()
# S1_5=S1_5/S1_5.max()
# filename = 'S1_0.dat'
# large_array_memmap=np.memmap(filename, dtype='float32', mode='w+', shape=S1_0.shape)
# large_array_memmap[:] = S1_0[:]
# filename = 'S1_1.dat'
# large_array_memmap=np.memmap(filename, dtype='float32', mode='w+', shape=S1_1.shape)
# large_array_memmap[:] = S1_1[:]
# filename = 'S1_2.dat'
# large_array_memmap=np.memmap(filename, dtype='float32', mode='w+', shape=S1_2.shape)
# large_array_memmap[:] = S1_2[:]
# filename = 'S1_3.dat'
# large_array_memmap=np.memmap(filename, dtype='float32', mode='w+', shape=S1_3.shape)
# large_array_memmap[:] = S1_3[:]
# filename = 'S1_4.dat'
# large_array_memmap=np.memmap(filename, dtype='float32', mode='w+', shape=S1_4.shape)
# large_array_memmap[:] = S1_4[:]
# filename = 'S1_5.dat'
# large_array_memmap=np.memmap(filename, dtype='float32', mode='w+', shape=S1_5.shape)
# large_array_memmap[:] = S1_5[:]

# S_s=[S1_0,S1_1,S1_2,S1_3,S1_4,S1_5]

#S1_0 = np.memmap('S1_0.dat', dtype='float32', mode='r', shape=(135, 189, 155))
#S1_1 = np.memmap('S1_1.dat', dtype='float32', mode='r', shape=(135, 189, 155))
#S1_2 = np.memmap('S1_2.dat', dtype='float32', mode='r', shape=(135, 189, 155))
#S1_3 = np.memmap('S1_3.dat', dtype='float32', mode='r', shape=(135, 189, 155))
#S1_4 = np.memmap('S1_4.dat', dtype='float32', mode='r', shape=(135, 189, 155))
#S1_5 = np.memmap('S1_5.dat', dtype='float32', mode='r', shape=(135, 189, 155))
S_s=['S1_0.dat','S1_1.dat','S1_2.dat','S1_3.dat','S1_4.dat','S1_5.dat']
x1_min_dom=-2*155/189
x1_max_dom=2*155/189
x2_min_dom=-2
x2_max_dom=2
x3_min_dom=-2*135/189
x3_max_dom=2*135/189
x1_min=0
x1_max_S1=S1_binary.shape[2]
x1_max_S2=S1_binary.shape[2]
x2_min=0
x2_max_S1=S1_binary.shape[1]
x2_max_S2=S1_binary.shape[1]
x3_min=0
x3_max_S1=S1_binary.shape[0]
x3_max_S2=S1_binary.shape[0]
id_co_x1_S1=(x1_max_S1-x1_min)/(x1_max_dom-x1_min_dom)
id_co_x2_S1=(x2_max_S1-x2_min)/(x2_max_dom-x2_min_dom)
id_co_x3_S1=(x3_max_S1-x3_min)/(x3_max_dom-x3_min_dom)
id_co_x1_S2=(x1_max_S2-x1_min)/(x1_max_dom-x1_min_dom)
id_co_x2_S2=(x2_max_S2-x2_min)/(x2_max_dom-x2_min_dom)
id_co_x3_S2=(x3_max_S2-x3_min)/(x3_max_dom-x3_min_dom)
co_tr_S1=np.array([[id_co_x1_S1,x1_min-id_co_x1_S1*x1_min_dom],[id_co_x2_S1,x2_min-id_co_x2_S1*x2_min_dom],
                [id_co_x3_S1,x3_min-id_co_x3_S1*x3_min_dom]])
co_tr_S2=np.array([[id_co_x1_S2,x1_min-id_co_x1_S2*x1_min_dom],[id_co_x2_S2,x2_min-id_co_x2_S2*x2_min_dom],
                [id_co_x3_S2,x3_min-id_co_x3_S2*x3_min_dom]])
layers=[3,60,60,60,3]
image_boundaries=[]
image_boundaries.append(ODE_Euler_end_time(XB,NN_t_pres[0]))
n_boundaries=5
for i in range(1,n_boundaries):
    image_boundaries.append(ODE_Euler_end_time(image_boundaries[i-1],NN_t_pres[i]))
#if not (S1_0.shape==S1_1.shape) & (S1_0.shape==S1_2.shape) & (S1_0.shape==S1_3.shape)&\
#      (S1_0.shape==S1_4.shape)  & (S1_0.shape==S1_5.shape)is True:
#  raise TypeError("images sizes must be the same size")
n_i=250
muu=1
lam=1
kappa=-5
center=0
n_mesh_x=45
n_mesh_y=45
n_mesh_z=45
batch_size_x=1500
x1_min_D=-2.5*155/189
x1_max_D=2.5*155/189
x2_min_D=-2.5
x2_max_D=2.5
x3_min_D=-2.5*135/189
x3_max_D=2.5*135/189
x =np.linspace(x1_min_dom,x1_max_dom,n_mesh_x)
y =np.linspace(x2_min_dom,x2_max_dom,n_mesh_y)
z=np.linspace(x3_min_dom,x3_max_dom,n_mesh_z)
dx=x[1]-x[0]
dy=y[1]-y[0]
dz=z[1]-z[0]
X1,X2=weights_coordinates(x,y)
X1,X2=gausspoints(X1,X2)
X1,X2=column_coef(X1,X2,n_mesh_x,n_mesh_y)
X1=np.reshape(X1,(int(X1.shape[0]/2),2,2))
X2=np.reshape(X2,(int(X2.shape[0]/2),2,2))
X1,X2,X3=column_coef_3D(X1,X2,z,n_mesh_x,n_mesh_y,n_mesh_z)
X1=jnp.array(X1)
X2=jnp.array(X2)
X3=jnp.array(X3)
x1_C=X1
x2_C=X2
x3_C=X3
#X1_O=X1
#X2_O=X2
#X3_O=X3
idn=np.arange(X1.shape[0])
idn=idn[0:np.shape(idn)[0]:2].tolist()
size = X1.size
X1_whole = jnp.reshape(X1, (size, 1))
X2_whole = jnp.reshape(X2, (size, 1))
X3_whole = jnp.reshape(X3, (size, 1))
x1x2x3_i_1=jnp.hstack((X1_whole,X2_whole,X3_whole))
del X1_whole, X2_whole, X3_whole
F_i_1= jnp.hstack((jnp.ones_like(x1x2x3_i_1[:,[0]]), jnp.zeros_like(x1x2x3_i_1[:,[0]]),jnp.zeros_like(x1x2x3_i_1[:,[0]]),\
                       jnp.zeros_like(x1x2x3_i_1[:,[1]]), jnp.ones_like(x1x2x3_i_1[:,[1]]),jnp.zeros_like(x1x2x3_i_1[:,[1]]),\
                        jnp.zeros_like(x1x2x3_i_1[:,[2]]),jnp.zeros_like(x1x2x3_i_1[:,[2]]),jnp.ones_like(x1x2x3_i_1[:,[2]])))
S1_binary=s1(x1x2x3_i_1,S1_binary).reshape((X1.shape[0],X1.shape[1],X1.shape[2]))
idn=np.arange(X1.shape[0])
idn=idn[0:np.shape(idn)[0]:2].tolist()
NN_t_Cs=[]
loss_totals_C=[]
loss_mismatches_C=[]
loss_energies_C=[]
loss_growths_C=[]
start = time.time()
for i in range(5):
        #D_grad_data=[jnp.ones((250, 250, 250)),jnp.ones((250, 250, 250)),jnp.ones((250, 250, 250))]
        if i==0:
           epoch_s=100000
        else:
           epoch_s=50000
        S1=np.memmap(S_s[i], dtype='float32', mode='r', shape=(135, 189, 155))
        S2=np.memmap(S_s[i+1], dtype='float32', mode='r', shape=(135, 189, 155))
        file_path = "/home/aamirihe/brain_step_pre_corrector/Gradx1.dat"
        if os.path.exists(file_path):
            os.remove(file_path)
        file_path = "/home/aamirihe/brain_step_pre_corrector/Gradx2.dat"
        if os.path.exists(file_path):
            os.remove(file_path)
        file_path = "/home/aamirihe/brain_step_pre_corrector/Gradx3.dat"
        if os.path.exists(file_path):
            os.remove(file_path)
        D_grad_data=Dgrad_calculations(image_boundaries[i])
        filename = 'Gradx1.dat'
        large_array_memmap=np.memmap(filename, dtype='float32', mode='w+', shape=D_grad_data[0].shape)
        large_array_memmap[:] = D_grad_data[0][:]
        filename = 'Gradx2.dat'
        large_array_memmap_2=np.memmap(filename, dtype='float32', mode='w+', shape=D_grad_data[1].shape)
        large_array_memmap_2[:] = D_grad_data[1][:]
        filename = 'Gradx3.dat'
        large_array_memmap_3=np.memmap(filename, dtype='float32', mode='w+', shape=D_grad_data[2].shape)
        large_array_memmap_3[:] = D_grad_data[2][:]
        del large_array_memmap_3, large_array_memmap_2,large_array_memmap, D_grad_data
        D_grad_data=[np.memmap('Gradx1.dat', dtype='float32', mode='r', shape=(250, 250, 250)),\
                     np.memmap('Gradx2.dat', dtype='float32', mode='r', shape=(250, 250, 250)),\
                     np.memmap('Gradx3.dat', dtype='float32', mode='r', shape=(250, 250, 250))]
        NN_C=init_params_b_shrinkage(layers, key,kappa)
        x1x2x3_i_1,F_i_1=predictor_and_gradiants(x1x2x3_i_1,F_i_1, NN_t_pres[i]) 
        x1x2x3_Cs=x1x2x3_i_1
        F_Cs=F_i_1
        x1x2x3_Cs=np.hsplit(x1x2x3_Cs,3); F_Cs=np.hsplit(F_Cs,9)
        x1x2x3_Cs[0]=x1x2x3_Cs[0].reshape((X1.shape[0], X1.shape[1],X1.shape[2])); x1x2x3_Cs[1]=x1x2x3_Cs[1].reshape((X1.shape[0], X1.shape[1],X1.shape[2]))
        x1x2x3_Cs[2]=x1x2x3_Cs[2].reshape((X1.shape[0], X1.shape[1],X1.shape[2]))
        F_Cs[0]=F_Cs[0].reshape((X1.shape[0], X1.shape[1],X1.shape[2]));F_Cs[1]=F_Cs[1].reshape((X1.shape[0], X1.shape[1],X1.shape[2]))
        F_Cs[2]=F_Cs[2].reshape((X1.shape[0], X1.shape[1],X1.shape[2]));F_Cs[3]=F_Cs[3].reshape((X1.shape[0], X1.shape[1],X1.shape[2]))
        F_Cs[4]=F_Cs[4].reshape((X1.shape[0], X1.shape[1],X1.shape[2]));F_Cs[5]=F_Cs[5].reshape((X1.shape[0], X1.shape[1],X1.shape[2]))
        F_Cs[6]=F_Cs[6].reshape((X1.shape[0], X1.shape[1],X1.shape[2]));F_Cs[7]=F_Cs[7].reshape((X1.shape[0], X1.shape[1],X1.shape[2]))
        F_Cs[8]=F_Cs[8].reshape((X1.shape[0], X1.shape[1],X1.shape[2]))
        [final_data,loss_total_1,Loss_mismatch_1,Loss_energy_1,Loss_growth_1,id_non]=\
             training(NN_C,x1_C,x2_C,x3_C,x1x2x3_Cs,F_Cs,batch_size_x,idn,image_boundaries[i],D_grad_data,S1,S2,S1_binary,.00005,epoch_s)
        loss_total=np.array(loss_total_1)[:,np.newaxis]; loss_mismatch=np.array(Loss_mismatch_1)[:,np.newaxis]
        loss_energy=np.array(Loss_energy_1)[:,np.newaxis]; loss_growth=np.array(Loss_growth_1)[:,np.newaxis]
        NN_t_Cs.append(final_data)
        loss_totals_C.append(loss_total)
        loss_mismatches_C.append(loss_mismatch)
        loss_energies_C.append(loss_energy)
        loss_growths_C.append(loss_growth)
        x1x2x3_dummy_i_1=np.zeros_like(x1x2x3_i_1)
        F_dummy_i_1=np.zeros_like(F_i_1)
        id_cor=x1x2x3_i_1.shape[0]//1500
        for j in range(id_cor):
          x1x2x3_dummy_i_1[j*1500:(j+1)*1500,:],F_dummy_i_1[j*1500:(j+1)*1500,:]=xs_and_Fs(x1x2x3_i_1[j*1500:(j+1)*1500,:],F_i_1[j*1500:(j+1)*1500,:],image_boundaries[i],D_grad_data,\
          final_data)
        x1x2x3_dummy_i_1[id_cor*1500:x1x2x3_i_1.shape[0]+1,:], F_dummy_i_1[id_cor*1500:x1x2x3_i_1.shape[0]+1,:]=\
        xs_and_Fs(x1x2x3_i_1[id_cor*1500:x1x2x3_i_1.shape[0]+1,:],F_i_1[id_cor*1500:x1x2x3_i_1.shape[0]+1,:],image_boundaries[i],D_grad_data,\
          final_data)
        print(j)
        print(x1x2x3_i_1.shape[0])
        x1x2x3_i_1=jnp.array(x1x2x3_dummy_i_1)
        F_i_1=jnp.array(F_dummy_i_1)
        del x1x2x3_dummy_i_1, F_dummy_i_1
        kappa= final_data[2]
        
end=time.time()
print(end-start)
with open('NN_t_Cs.pickle', 'wb') as f:
    pickle.dump(NN_t_Cs, f)
with open('Total_Loss_corrector_fig8.pickle', 'wb') as f:
    pickle.dump(loss_totals_C, f)
with open('Mismatch_Loss_corrector_fig8.pickle', 'wb') as f:
    pickle.dump(loss_mismatches_C, f)
with open('Energy_Loss_corrector_fig8.pickle', 'wb') as f:
    pickle.dump(loss_energies_C, f)
with open('Loss_growth_fig8.pickle', 'wb') as f:
    pickle.dump(loss_growths_C, f)