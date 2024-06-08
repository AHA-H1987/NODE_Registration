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
from jax import grad, jit, vmap,jacfwd,jacrev
from functools import partial
import optax
from jax.nn import gelu
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
####
#######initialize parameters
def init_params_b_growth(layers, key,k):
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
@jit
def image_boundary (x,x_min:int,x_max:int):
    id_boundary=jnp.where((x_min<x)&(x<x_max) ,1,0)
    id_boundary1=jnp.round(x* id_boundary)
    #x=torch.where(id_boundary==0,torch.ones_like(x),x)
    #x
    return id_boundary,id_boundary1
@jit
def s2(x1x2,img):
    x1 = x1x2[:, [0]] * co_tr[0, 0] + co_tr[0, 1]
    x2 = x1x2[:, [1]] * co_tr[1, 0] + co_tr[1, 1]
    x1 = x1
    x2 =x2_max- x2
    x1_floor = jnp.floor(x1).astype(int)
    x1_ceil = x1_floor + 1
    i01, i1 = image_boundary(x1_floor, x1_min, x1_max)
    i02, i2= image_boundary(x1_ceil, x1_min, x1_max)
    x2_floor = jnp.floor(x2).astype(int)
    x2_ceil = x2_floor + 1
    j01, j1 = image_boundary(x2_floor, x2_min, x2_max)
    j02, j2= image_boundary(x2_ceil, x2_min, x2_max)
    ans = j01 * i01 * img[j1, i1] * (x1 - x1_ceil) * (x2 - x2_ceil) / (
            (x1_floor - x1_ceil) * (x2_floor - x2_ceil)) + \
          j01 * i02 * img[j1, i2] * (x1 - x1_floor) * (x2 - x2_ceil) / (
                  (x1_ceil - x1_floor) * (x2_floor - x2_ceil)) + \
          j02 * i02 * img[j2, i2]* (x1 - x1_floor) * (x2 - x2_floor) / (
                  (x1_ceil - x1_floor) * (x2_ceil - x2_floor)) + \
          j02 * i01 *img[j2, i1] * (x1 - x1_ceil) * (x2 - x2_floor) / (
                  (x1_floor - x1_ceil) * (x2_ceil - x2_floor))
    return ans
@jit
def s1_binary(X1X2,img):
    x1_min=0
    x1_max=S1_binary.shape[1]
    x2_min=0
    x2_max=S1_binary.shape[0]
    id_co_x1=(x1_max-x1_min)/(x1_max_dom-x1_min_dom)
    id_co_x2=(x2_max-x2_min)/(x2_max_dom-x2_min_dom)
    co_tr=np.array([[id_co_x1,x1_min-id_co_x1*x1_min_dom],[id_co_x2,x2_min-id_co_x2*x2_min_dom]])
    X1=X1X2[:,[0]]*co_tr[0,0]+co_tr[0,1]
    X2=X1X2[:, [1]] * co_tr[1, 0] + co_tr[1, 1]
    x1=X1
    x2=x2_max-X2
    x1_floor = jnp.floor(x1).astype(int)
    x1_ceil = x1_floor + 1
    i01, i1= image_boundary(x1_floor, x1_min, x1_max)
    i02, i2 = image_boundary(x1_ceil, x1_min, x1_max)
    x2_floor = jnp.floor(x2).astype(int)
    x2_ceil = x2_floor + 1
    j01, j1= image_boundary(x2_floor, x2_min, x2_max)
    j02, j2= image_boundary(x2_ceil, x2_min, x2_max)
    ans = j01*i01*img[j1, i1] * (x1 - x1_ceil) * (x2 - x2_ceil) / (
            (x1_floor - x1_ceil) * (x2_floor - x2_ceil)) + \
          j01*i02*img[j1, i2] * (x1 - x1_floor) * (x2 - x2_ceil) / (
                  (x1_ceil - x1_floor) * (x2_floor - x2_ceil)) + \
          j02*i02*img[j2, i2] * (x1 - x1_floor) * (x2 - x2_floor) / (
                  (x1_ceil - x1_floor) * (x2_ceil - x2_floor)) + \
          j02*i01*img[j2, i1] * (x1 - x1_ceil) * (x2 - x2_floor) / (
                  (x1_floor - x1_ceil) * (x2_ceil - x2_floor))
    return ans
@jit
def s1(X1X2,img):
    X1=X1X2[:,[0]]*co_tr[0,0]+co_tr[0,1]
    X2=X1X2[:, [1]] * co_tr[1, 0] + co_tr[1, 1]
    x1=X1
    x2=x2_max-X2
    x1_floor = jnp.floor(x1).astype(int)
    x1_ceil = x1_floor + 1
    i01, i1= image_boundary(x1_floor, x1_min, x1_max)
    i02, i2 = image_boundary(x1_ceil, x1_min, x1_max)
    x2_floor = jnp.floor(x2).astype(int)
    x2_ceil = x2_floor + 1
    j01, j1= image_boundary(x2_floor, x2_min, x2_max)
    j02, j2= image_boundary(x2_ceil, x2_min, x2_max)
    ans = j01*i01*img[j1, i1] * (x1 - x1_ceil) * (x2 - x2_ceil) / (
            (x1_floor - x1_ceil) * (x2_floor - x2_ceil)) + \
          j01*i02*img[j1, i2] * (x1 - x1_floor) * (x2 - x2_ceil) / (
                  (x1_ceil - x1_floor) * (x2_floor - x2_ceil)) + \
          j02*i02*img[j2, i2] * (x1 - x1_floor) * (x2 - x2_floor) / (
                  (x1_ceil - x1_floor) * (x2_ceil - x2_floor)) + \
          j02*i01*img[j2, i1] * (x1 - x1_ceil) * (x2 - x2_floor) / (
                  (x1_floor - x1_ceil) * (x2_ceil - x2_floor))
    return ans
@jit
def dNN_dxs(x1x2,nn_f):
    dNN1_dxs_= vmap(grad(lambda x1x2_: forward_pass_b(x1x2_, nn_f)[0]))
    dNN2_dxs_= vmap(grad(lambda x1x2_: forward_pass_b(x1x2_, nn_f)[1]))
    return jnp.hstack((dNN1_dxs_(x1x2),dNN2_dxs_(x1x2)))
@jit
def mesh_address(x,x_min,x_max,n):
    params=jnp.array([-n*x_min/(x_max-x_min),n/(x_max-x_min)])
    map=params[0]+params[1]*x
    return [jnp.floor(map),jnp.floor(map)+1]
def ODE_Euler_end_time(X1X2,nn_f):
    x1x2=X1X2
    for i in range(time_steps):
        x1x2_change=forward_pass_b(x1x2,nn_f)
        x1x2=x1x2+ sudo_dt * x1x2_change
    return x1x2
@jit
def D_function(x1x2,image_boundary_co):
    #x1x2=jnp.where(( x1x2[0]>x1_max) | (x1x2[0]<x1_min)|(x1x2[1]>x2_max) | (x1x2[1]<x2_min), jnp.zeros_like(x1x2),x1x2)
    coo_x1 = image_boundary_co[:, [0]] - x1x2[0]
    coo_x2 = image_boundary_co[:, [1]] - x1x2[1]
    return jnp.min(jnp.sqrt(coo_x1 ** 2 + coo_x2 ** 2))
def Dgrad_calculations(image_boundary):
    x1_min=-2.5*610/592
    x1_max=2.5*610/592
    x2_min=-2.5
    x2_max=2.5
    n_mesh_x=250
    n_mesh_y=250
    x1 =np.linspace(x1_min,x1_max,n_mesh_x)
    x2 =np.linspace(x2_min,x2_max,n_mesh_y)
    x1,x2=np.meshgrid(x1,x2,indexing='xy')
    x1_shape=x1.shape
    x1 = jnp.reshape(x1, (x1.size, 1))
    x2= jnp.reshape(x2, (x2.size, 1))
    x1x2=jnp.hstack((x1,x2))
    DGrad_function_vmap=vmap(jacfwd(lambda  x1x2_:D_function( x1x2_,image_boundary)))
    ans=np.zeros((1,2))
    for i in range(1000):
        x1x2x3_var=x1x2[125*125*i:125*125*i+125*125,:]
        ans=np.append(ans,np.array(DGrad_function_vmap(x1x2x3_var)),axis=0)
    ans=ans[1:,:]
    # ans2=D_function_vmap(x1x2x3)
    Grad_x1=jnp.reshape(ans[:,[0]],x1_shape)
    Grad_x2=jnp.reshape(ans[:,[1]],x1_shape)
    D_Grad_data=[Grad_x1,Grad_x2]
    print(np.array(Grad_x1).shape)
    print(np.array(Grad_x2).shape)
    print(np.array(Grad_x1).max())
    print(np.array(Grad_x2).max())
    return D_Grad_data
@jit
def DGrad_interpolation(x1x2,x1_min,x1_max,x2_min,x2_max,n,data):
    data_x1=data[0]
    data_x2=data[1]
    x1x2=jnp.where(( x1x2[:,[0]]>x1_max) | (x1x2[:,[0]]<x1_min)|(x1x2[:,[1]]>x2_max) | (x1x2[:,[1]]<x2_min), jnp.zeros_like(x1x2),x1x2)
    x1=x1x2[:,[0]]
    x2=x1x2[:,[1]]
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
    ans1= data_x1[nx2_floor, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_ceil) / (
            (x1_floor - x1_ceil) * (x2_floor - x2_ceil)) + \
          data_x1[nx2_floor, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_ceil) / (
                  (x1_ceil - x1_floor) * (x2_floor - x2_ceil)) + \
          data_x1[nx2_ceil, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_floor) / (
                  (x1_ceil - x1_floor) * (x2_ceil - x2_floor)) + \
          data_x1[nx2_ceil, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_floor) / (
                  (x1_floor - x1_ceil) * (x2_ceil - x2_floor))
    ans2 = data_x2[nx2_floor, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_ceil) / (
            (x1_floor - x1_ceil) * (x2_floor - x2_ceil)) + \
          data_x2[nx2_floor, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_ceil) / (
                  (x1_ceil - x1_floor) * (x2_floor - x2_ceil)) + \
          data_x2[nx2_ceil, nx1_ceil] * (x1 - x1_floor) * (x2 - x2_floor) / (
                  (x1_ceil - x1_floor) * (x2_ceil - x2_floor)) + \
          data_x2[nx2_ceil, nx1_floor] * (x1 - x1_ceil) * (x2 - x2_floor) / (
                  (x1_floor - x1_ceil) * (x2_ceil - x2_floor))
    return ans1,ans2
@jit
def xs_and_Fs(x1x2_p_mini,Fp_values_mini,XB_i,D_grad_data_i,nn_f):
    Fs_c=Fp_values_mini
    eye=jnp.array([1,0,0,1])
    x1x2=x1x2_p_mini
    for i in range(time_steps):
        D_function_vmap=vmap(lambda x1x2_:D_function(x1x2_,XB_i))
        D_values_mini = D_function_vmap(x1x2)[:,None]
        D_Gradx1_values_mini, D_Gradx2_values_mini = DGrad_interpolation(x1x2, x1_min_D, x1_max_D, x2_min_D,  x2_max_D, n_i, D_grad_data_i)
        g1g2_values = forward_pass_b(x1x2, nn_f)
        dNN_dxs_values = dNN_dxs(x1x2, nn_f)
        g_times_dD_dX = jnp.hstack((g1g2_values[:, [0]] * D_Gradx1_values_mini, g1g2_values[:, [0]] * D_Gradx2_values_mini,\
                                    g1g2_values[:, [1]] * D_Gradx1_values_mini, g1g2_values[:, [1]] * D_Gradx2_values_mini ))
        Fs_c_n=eye+ sudo_dt * g_times_dD_dX  +sudo_dt *D_values_mini *dNN_dxs_values
        Fs_c=jnp.hstack((Fs_c_n[:,[0]]*Fs_c[:,[0]]+Fs_c_n[:,[1]]*Fs_c[:,[2]],\
                         Fs_c_n[:,[0]]*Fs_c[:,[1]]+Fs_c_n[:,[1]]*Fs_c[:,[3]],\
                         Fs_c_n[:,[2]]*Fs_c[:,[0]]+Fs_c_n[:,[3]]*Fs_c[:,[2]],\
                         Fs_c_n[:,[2]]*Fs_c[:,[1]]+Fs_c_n[:,[3]]*Fs_c[:,[3]]))
        x1x2=x1x2+ sudo_dt * D_values_mini*forward_pass_b(x1x2,nn_f)
    return x1x2,Fs_c
def x_value(X1X2,nn_f):
    x1x2=X1X2
    for i in range(time_steps):
        x1x2=x1x2+ sudo_dt * forward_pass_b(x1x2,nn_f)
    return x1x2
def F_values(X1X2,nn_f):
    dx1dX=vmap(grad(lambda x1x2: x_value(x1x2, nn_f)[0]))
    dx2dX = vmap(grad(lambda x1x2:x_value(x1x2, nn_f)[1]))
    F11_12=dx1dX(X1X2)
    F21_22=dx2dX(X1X2)
    return jnp.hstack((F11_12[:,[0]],F11_12[:,[1]],F21_22[:,[0]],F21_22[:,[1]]))
def predictor_and_gradiants(X1X2, F0,nn_f_p):
    F_values_p=F_values(X1X2, nn_f_p)
    F_values_p_material= jnp.hstack((F_values_p[:,[0]] * F0[:, [0]] + F_values_p[:,[1]]  * F0[:, [2]],\
                                     F_values_p[:,[0]] * F0[:, [1]] + F_values_p[:,[1]]  * F0[:, [3]],\
                                     F_values_p[:,[2]] * F0[:, [0]] + F_values_p[:,[3]]  * F0[:, [2]], \
                                     F_values_p[:,[2]] * F0[:, [1]] + F_values_p[:,[3]]  * F0[:, [3]]))
    x1x2_p=ODE_Euler_end_time(X1X2,nn_f_p)
    return [ x1x2_p, F_values_p_material]
@jit
def psi( Fs,X1X2,S1_mesh,sf):
    s1_X=s1_binary(X1X2,S1_mesh)
    Fs=Fs*jnp.exp(-jnp.exp(sf))
    F11 = Fs[:, [0]]
    F12 = Fs[:, [1]]
    F21 = Fs[:, [2]]
    F22 = Fs[:, [3]]
    C11 = F11 * F11 + F21 * F21
    C12 = F11 * F12 + F21 * F22
    C21 = C12
    C22 = F12 * F12 + F22 * F22
    trace_C = C11 + C22
    det_F = F11 * F22 - F12 * F21
    det_F=jnp.where(s1_X==1,det_F,jnp.ones_like(det_F))
    trace_C=jnp.where(s1_X==1,trace_C,2*jnp.ones_like(trace_C))
    energy = (.5 * muu * (trace_C + 1 - 3) - muu * jnp.log(det_F) + .5 * lam * (jnp.log(det_F)) ** 2)
    return energy
@jit
def s1_s2(X1X2,x1x2,S1_image,S2_image):
    s1_b=s1_binary(X1X2,S1_binary)
    s1_X =s1(X1X2,S1_image)
    #s1_X = jnp.where(jnp.abs(s1_X - 1)<.0001, s1_X, jnp.zeros_like(s1_X))
    s1_X = jnp.where(s1_b==1, s1_X, jnp.zeros_like(s1_X))
    s2_phi_X =s2(x1x2,S2_image)
    s2_phi_X = jnp.where(s1_b == 1, s2_phi_X, jnp.zeros_like(s2_phi_X))
    ans = (s1_X - s2_phi_X) ** 2
    return ans
@jit
def integral(nn_f,X1X2,x1x2_p_mini,Fp_values_mini,XB_i,D_grad_data_i,S1_image,S2_image):
    x1x2,Fs=xs_and_Fs(x1x2_p_mini,Fp_values_mini,XB_i,D_grad_data_i,nn_f)
    coef=jnp.ones_like(X1X2[:,[0]]) * dx * dy / 4
    var0=s1_s2(X1X2,x1x2,S1_image,S2_image)
    var1=psi(Fs,X1X2,S1_binary,nn_f[2])
    second_integrand=s1_binary(X1X2,S1_binary)* var1#S1_gauss
    first_integral=jnp.trace(jnp.matmul(jnp.transpose(var0), coef))
    second_integral=jnp.trace(jnp.matmul(jnp.transpose(second_integrand), coef))
    return [second_integral, first_integral, second_integral,[],[]]
@jit
def loss_computation(nn_f,X1X2,x1x2_p_mini,Fp_values_mini,XB_i,D_grad_data_i,S1_image,S2_image):
    vrbl = integral(nn_f,X1X2,x1x2_p_mini,Fp_values_mini,XB_i,D_grad_data_i,S1_image,S2_image)
    loss0 = vrbl[0]
    return [loss0, vrbl[1], vrbl[2], vrbl[3], vrbl[4]]
def training(nn_f,X1,X2,x1x2_p,Fp_values,batch_size_x,idn,XB_i,D_grad_data_i,S1_image,S2_image,lr, epoch_max:int=1000):
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
        size = X1_mini.shape[0] * X1_mini.shape[1]
        X1_mini = jnp.reshape(X1_mini, (size, 1))
        X2_mini = jnp.reshape(X2_mini, (size, 1))
        X1X2_mini=jnp.hstack((X1_mini,X2_mini))
        x1x2_p_mini = jnp.hstack((jnp.squeeze(x1x2_p[0][intrand, :]).reshape((size, 1)), jnp.squeeze(x1x2_p[1][intrand, :]).reshape((size, 1))))
        Fp_values_mini = jnp.hstack((jnp.squeeze(Fp_values[0][intrand, :]).reshape((size, 1)), jnp.squeeze(Fp_values[1][intrand, :]).reshape((size, 1)), \
                                     jnp.squeeze(Fp_values[2][intrand, :]).reshape((size, 1)), jnp.squeeze(Fp_values[3][intrand, :]).reshape((size, 1))))
        try:
           a=loss_computation(nn_f,X1X2_mini,x1x2_p_mini,Fp_values_mini,XB_i,D_grad_data_i,S1_image,S2_image)
           loss=a[0]
           if jnp.isnan(loss):
               id_non=id_non+1
               print(id)
           #print(a[0])
           if ~(jnp.isnan(loss) | jnp.isinf(loss)):
               grad_for_opt=lambda nn_f: loss_computation(nn_f,X1X2_mini,x1x2_p_mini,Fp_values_mini,XB_i,D_grad_data_i,S1_image,S2_image)[0]
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
        x_bar = np.vstack((x_bar, x[i - 1:i + 1]))
    for i in range(2, n_mesh_y):
        y_bar = np.vstack((y_bar, y[i - 1:i + 1]))
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
def img_extension(img,m):
    img=img[120:,40:450]
    #new_img=np.zeros(im_0[0:5,:].min(),im_0[0:5,:].max(),(img.shape[0]+2*m,img.shape[1]+2*m))
    new_img=np.zeros((img.shape[0]+2*m,img.shape[1]+2*m))
    new_img[m:m+img.shape[0],m:m+img.shape[1]]=img
    return new_img
### program starts from here
with open("MAX_trial001_2.txt") as BP:
    ad=0
    for i in BP:
        if ad==0:
            j = i.split()
            im_5 = [[float(e) for e in j]]
        else:
            j = i.split()
            num=[[float(e) for e in j]]
            im_5= np.append(im_5, num, axis=0)
        ad=1
with open("MAX_trial001_22.txt") as BP:
    ad=0
    for i in BP:
        if ad==0:
            j = i.split()
            im_4 = [[float(e) for e in j]]
        else:
            j = i.split()
            num=[[float(e) for e in j]]
            im_4= np.append(im_4, num, axis=0)
        ad=1
with open("MAX_trial001_42.txt") as BP:
    ad=0
    for i in BP:
        if ad==0:
            j = i.split()
            im_3 = [[float(e) for e in j]]
        else:
            j = i.split()
            num=[[float(e) for e in j]]
            im_3= np.append(im_3, num, axis=0)
        ad=1
with open("MAX_trial001_62.txt") as BP:
    ad=0
    for i in BP:
        if ad==0:
            j = i.split()
            im_2 = [[float(e) for e in j]]
        else:
            j = i.split()
            num=[[float(e) for e in j]]
            im_2= np.append(im_2, num, axis=0)
        ad=1
with open("MAX_trial001_82.txt") as BP:
    ad=0
    for i in BP:
        if ad==0:
            j = i.split()
            im_1 = [[float(e) for e in j]]
        else:
            j = i.split()
            num=[[float(e) for e in j]]
            im_1= np.append(im_1, num, axis=0)
        ad=1
with open("MAX_trial001_102.txt") as BP:
    ad=0
    for i in BP:
        if ad==0:
            j = i.split()
            im_0 = [[float(e) for e in j]]
        else:
            j = i.split()
            num=[[float(e) for e in j]]
            im_0= np.append(im_0, num, axis=0)
        ad=1
with open("MASK_MAX_trial001_main_binary-102.txt") as BP:
    ad=0
    for i in BP:
        if ad==0:
            j = i.split()
            S1_binary = [[float(e) for e in j]]
        else:
            j = i.split()
            num=[[float(e) for e in j]]
            S1_binary= np.append(S1_binary, num, axis=0)
        ad=1
S1_binary=S1_binary/S1_binary.max()
S1_binary=img_extension(S1_binary,100)
with open('BP.pickle', 'rb') as f:
   XB= pickle.load(f)
with open('NN_t_pre_fig6.pickle', 'rb') as f:
    NN_pres= pickle.load(f)
im_5=im_5/im_5.max()
im_4=im_4/im_4.max()
im_3=im_3/im_3.max()
im_2=im_2/im_2.max()
im_1=im_1/im_1.max()
im_0=im_0/im_0.max()
im_0=img_extension(im_0,100)
im_1=img_extension(im_1,100)
im_2=img_extension(im_2,100)
im_3=img_extension(im_3,100)
im_4=img_extension(im_4,100)
im_5=img_extension(im_5,100)
S1_images=[im_0,im_1,im_2,im_3,im_4]
S2_images=[im_1,im_2,im_3,im_4,im_5]
timeframe=5
time_steps=15
sudo_dt=1/time_steps
n_i=250
image_boundaries=[]
size_b=XB.shape[0]
#d_b.append(jnp.zeros((size_b,1)))
image_boundaries.append(ODE_Euler_end_time(XB,NN_pres[0]))
n_boundaries=5
for i in range(1,n_boundaries):
    print(i)
    image_boundaries.append(ODE_Euler_end_time(image_boundaries[i-1],NN_pres[i]))
if not (im_0.shape==im_1.shape) & (im_0.shape==im_2.shape) & (im_0.shape==im_3.shape)&\
      (im_0.shape==im_4.shape)  & (im_0.shape==im_5.shape)is True:
  raise TypeError("images sizes must be the same size")
x1_min_dom=-2*610/592
x1_max_dom=2*610/592
x2_min_dom=-2
x2_max_dom=2
x1_min=0
x1_max=im_0.shape[1]
x2_min=0
x2_max=im_0.shape[0]
id_co_x1=(x1_max-x1_min)/(x1_max_dom-x1_min_dom)
id_co_x2=(x2_max-x2_min)/(x2_max_dom-x2_min_dom)
co_tr=np.array([[id_co_x1,x1_min-id_co_x1*x1_min_dom],[id_co_x2,x2_min-id_co_x2*x2_min_dom]])
layers=[2,40,40,40,2]
# im_size=55
# XB[:,[0]]=(x1_max_dom-x1_min_dom)*XB[:,[0]]/im_size+x1_min_dom
# XB[:,[1]]=(x2_max_dom-x2_min_dom)*XB[:,[1]]/im_size+x2_min_dom
# plt.scatter(image_boundary_co[:,[0]], image_boundary_co[:,[1]])
# plt.show()
x1_min_D=-2.5*610/592
x1_max_D=2.5*610/592
x2_min_D=-2.5
x2_max_D=2.5
kappa=-12
muu=1
lam=1
n_mesh_x=200
n_mesh_y=200
batch_size_x=2000
x =np.linspace(x1_min_dom,x1_max_dom,n_mesh_x)
y =np.linspace(x2_min_dom,x2_max_dom,n_mesh_y)
dx=x[1]-x[0]
dy=y[1]-y[0]
X1,X2=weights_coordinates(x,y)
X1,X2=gausspoints(X1,X2)
X1,X2=column_coef(X1,X2,n_mesh_x,n_mesh_y)
X1=np.array(X1)
X2=np.array(X2)
size = X1.shape[0] * X1.shape[1]
idn=np.arange(X1.shape[0])
idn=idn[0:np.shape(idn)[0]:2].tolist()
x1x2_i_1=jnp.hstack((X1.reshape((size, 1)),X2.reshape((size, 1))))
F_i_1= jnp.hstack((jnp.ones_like(X1.reshape((size, 1))), jnp.zeros_like(X1.reshape((size, 1))), \
                    jnp.zeros_like(X2.reshape((size, 1))), jnp.ones_like(X2.reshape((size, 1)))))
##############################################################################################################
NN_t_Cs=[]
loss_totals_C=[]
loss_mismatches_C=[]
loss_energies_C=[]
loss_growths_C=[]
start = time.time()
for i in range(timeframe):
    if i==(timeframe-1):
           epoch_s=100000
           kappa=-4
    else:
           epoch_s=50000
           kappa=-17
    if i==3:
        epoch_s=100000
        kappa=-15
    x1x2_i_1,F_i_1=predictor_and_gradiants(x1x2_i_1,F_i_1, NN_pres[i])
    x1x2_p=x1x2_i_1
    Fp_values=F_i_1
    x1x2_p=np.hsplit(x1x2_p,2); Fp_values=np.hsplit(Fp_values,4)
    x1x2_p[0]=x1x2_p[0].reshape((X1.shape[0], X1.shape[1])); x1x2_p[1]=x1x2_p[1].reshape((X1.shape[0], X1.shape[1]))
    Fp_values[0]=Fp_values[0].reshape((X1.shape[0], X1.shape[1]));Fp_values[1]=Fp_values[1].reshape((X1.shape[0], X1.shape[1]))
    Fp_values[2]=Fp_values[2].reshape((X1.shape[0], X1.shape[1]));Fp_values[3]=Fp_values[3].reshape((X1.shape[0], X1.shape[1]))
    D_grad_data=Dgrad_calculations(image_boundaries[i])
    NN_C=init_params_b_growth(layers, key,kappa)
    [final_data,loss_total_1,Loss_mismatch_1,Loss_energy_1,Loss_growth_1,id_non]=\
    training(NN_C,X1,X2,x1x2_p,Fp_values,batch_size_x,idn,image_boundaries[i],D_grad_data,S1_images[i],S2_images[i],.00005,epoch_s)
    loss_total=np.array(loss_total_1)[:,np.newaxis]; loss_mismatch=np.array(Loss_mismatch_1)[:,np.newaxis]
    loss_energy=np.array(Loss_energy_1)[:,np.newaxis]; loss_growth=np.array(Loss_growth_1)[:,np.newaxis]
    NN_t_Cs.append(final_data)
    loss_totals_C.append(loss_total)
    loss_mismatches_C.append(loss_mismatch)
    loss_energies_C.append(loss_energy)
    loss_growths_C.append(loss_growth)
    x1x2_i_1,F_i_1=xs_and_Fs(x1x2_i_1,F_i_1,image_boundaries[i],D_grad_data,NN_t_Cs[i])
with open('trained_NNs_c.pickle', 'wb') as f:
        pickle.dump(NN_t_Cs, f)
with open('loss_totals_c.pickle', 'wb') as f:
        pickle.dump(loss_totals_C, f)
with open('loss_mismatchs_c.pickle', 'wb') as f:
        pickle.dump(loss_mismatches_C, f)
with open('loss_energies_c.pickle', 'wb') as f:
        pickle.dump(loss_energies_C, f)
with open('loss_growth_c.pickle', 'wb') as f:
        pickle.dump(loss_growths_C, f)
end=time.time()
print(end-start)
