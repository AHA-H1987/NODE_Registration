import jax.numpy as jnp
import pickle
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
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

####

#######initialize parameters
def init_params_b(layers, key):
  Ws = []
  bs = []
  for i in range(len(layers) - 1):
    std_glorot = jnp.sqrt(2/(layers[i] + layers[i + 1]))
    key, subkey = jax.random.split(key)
    Ws.append(jax.random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    bs.append(jnp.zeros(layers[i + 1]))
  return (Ws, bs)
def init_params_b_zeros(layers, key):
  Ws = []
  bs = []
  for i in range(len(layers) - 1):
    Ws.append(jnp.zeros((layers[i], layers[i + 1])))
    bs.append(jnp.zeros(layers[i + 1]))
  return (Ws, bs)
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
def dNN_dxs(x1x2x3,nn_f):
    dNN1_dxs_= vmap(grad(lambda x1x2x3_: forward_pass_b(x1x2x3_, nn_f)[0]))
    dNN2_dxs_= vmap(grad(lambda x1x2x3_: forward_pass_b(x1x2x3_, nn_f)[1]))
    dNN3_dxs_= vmap(grad(lambda x1x2x3_: forward_pass_b(x1x2x3_, nn_f)[2]))
    return jnp.hstack((dNN1_dxs_(x1x2x3),dNN2_dxs_(x1x2x3), dNN3_dxs_(x1x2x3)))

@jit
def xs_and_Fs(X1X2X3,nn_f):
    x1x2x3=X1X2X3
    Fs_ = jnp.hstack((jnp.ones_like(x1x2x3[:,[0]]), jnp.zeros_like(x1x2x3[:,[0]]),jnp.zeros_like(x1x2x3[:,[0]]),\
                       jnp.zeros_like(x1x2x3[:,[1]]), jnp.ones_like(x1x2x3[:,[1]]),jnp.zeros_like(x1x2x3[:,[1]]),\
                        jnp.zeros_like(x1x2x3[:,[2]]),jnp.zeros_like(x1x2x3[:,[2]]),jnp.ones_like(x1x2x3[:,[2]])))
    for i in range(time_steps):
        dNN_dxs_values = dNN_dxs(x1x2x3, nn_f)
        F11_p = 1 + sudo_dt * dNN_dxs_values[:,[0]]
        F12_p = sudo_dt* dNN_dxs_values[:,[1]]
        F13_p = sudo_dt* dNN_dxs_values[:,[2]]
        F21_p = sudo_dt* dNN_dxs_values[:,[3]]
        F22_p = 1 + sudo_dt * dNN_dxs_values[:,[4]]
        F23_p = sudo_dt* dNN_dxs_values[:,[5]]
        F31_p = sudo_dt* dNN_dxs_values[:,[6]]
        F32_p = sudo_dt* dNN_dxs_values[:,[7]]
        F33_p = 1 + sudo_dt * dNN_dxs_values[:,[8]]
        # 11 0 12 1 13 2 21 3 22 4 23 5 31 6 32 7 33 8
        Fs_ = jnp.hstack((F11_p*Fs_[:, [0]] + F12_p*Fs_[:, [3]] + F13_p*Fs_[:, [6]],  F11_p*Fs_[:, [1]] + F12_p*Fs_[:, [4]] + F13_p*Fs_[:, [7]], F11_p*Fs_[:, [2]] + F12_p*Fs_[:, [5]] + F13_p*Fs_[:, [8]],
                          F21_p*Fs_[:, [0]] + F22_p*Fs_[:, [3]] + F23_p*Fs_[:, [6]], F21_p*Fs_[:, [1]] + F22_p*Fs_[:, [4]] + F23_p*Fs_[:, [7]], F21_p*Fs_[:, [2]] + F22_p*Fs_[:, [5]] + F23_p*Fs_[:, [8]],
                          F31_p*Fs_[:, [0]] + F32_p*Fs_[:, [3]] + F33_p*Fs_[:, [6]], F31_p*Fs_[:, [1]] + F32_p*Fs_[:, [4]] + F33_p*Fs_[:, [7]],  F31_p*Fs_[:, [2]] + F32_p*Fs_[:, [5]] + F33_p*Fs_[:, [8]]))
        x1x2x3=x1x2x3+ sudo_dt * forward_pass_b(x1x2x3,nn_f)
    return x1x2x3,Fs_

@jit
def image_boundary (x,x_min:int,x_max:int):
    id_boundary=jnp.where((x_min<x)&(x<x_max) ,1,0)
    id_boundary1=jnp.round(x* id_boundary)
    #x=torch.where(id_boundary==0,torch.ones_like(x),x)
    #x
    return id_boundary,id_boundary1
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
def psi( Fs):
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
    det_F = F11*F22*F33 - F11*F23*F32 - F12*F21*F33 + F12*F23*F31 + F13*F21*F32 - F13*F22*F31
    energy = (.5 * muu * (trace_C- 3) - muu * jnp.log(det_F) + .5 * lam * (jnp.log(det_F)) ** 2)
    return energy
@jit
def s1_s2(X1X2X3,x1x2x3):
    s1_X =s1(X1X2X3,S1)
    s2_phi_X =s2(x1x2x3,S2)
    ans = (s1_X - s2_phi_X) ** 2
    return ans
@jit
def integral(nn_f,X1X2X3):
    x1x2x3,Fs=xs_and_Fs(X1X2X3,nn_f)
    coef=jnp.ones_like(X1X2X3[:,[0]]) * dx * dy*dz / 8
    #surf_int=jnp.sum(coef)
    var0=s1_s2(X1X2X3,x1x2x3)
    var1=psi(Fs)
    second_integrand = s1(X1X2X3,S1)* var1
    first_integral=jnp.trace(jnp.matmul(jnp.transpose(var0), coef))
    second_integral = jnp.trace(jnp.matmul(jnp.transpose(second_integrand), coef))
    return [(1*first_integral + second_integral/10000), first_integral, second_integral,[],[]]
@jit
def loss_computation(nn_f,  X1X2X3):
    vrbl = integral(nn_f, X1X2X3)
    loss0 = vrbl[0]
    # loss1=torch.zeros(1,1)
    return [loss0, vrbl[1], vrbl[2], vrbl[3], vrbl[4]]
def training(nn_f,X1,X2,X3,batch_size_x,idn,lr, epoch_max:int=1000):
    loss_evolution = []
    loss_evolution_mismatch = []
    loss_evolution_energy = []
    # loss_evolution =np.squeeze(np.load('Total_Loss_pre_im_fig4.npy',allow_pickle=True)).tolist()
    # loss_evolution_mismatch =np.squeeze(np.load('Mismatch_Loss_pre_im_fig4.npy',allow_pickle=True)).tolist()
    # loss_evolution_energy = np.squeeze(np.load('Energy_Loss_pre_im_fig4.npy',allow_pickle=True)).tolist()
    loss_evolution_growth = []
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
        X1_mini = jnp.squeeze(X1[intrand, :,:])
        X2_mini = jnp.squeeze(X2[intrand, :,:])
        X3_mini = jnp.squeeze(X3[intrand, :,:])
        size = X1_mini.shape[0] * X1_mini.shape[1]*X1_mini.shape[2]
        X1_mini = jnp.reshape(X1_mini, (size, 1))
        X2_mini = jnp.reshape(X2_mini, (size, 1))
        X3_mini = jnp.reshape(X3_mini, (size, 1))
        X1X2X3_mini=jnp.hstack((X1_mini,X2_mini,X3_mini))
        try:
           #start = time.time()
           a=loss_computation(nn_f,X1X2X3_mini)
           loss=a[0]
           if ~(jnp.isnan(loss) | jnp.isinf(loss)):
               grad_for_opt=lambda nn_f: loss_computation(nn_f,X1X2X3_mini)[0]
               grads = jax.grad(grad_for_opt)(nn_f)
               updates, opt_state = optimizer.update(grads, opt_state)
               nn_f = optax.apply_updates(nn_f, updates)
           if i % 10==0:
                print(f'epoch={i}, loss={loss}')
                print(f"imag_mismatch={a[1]}")
                print(f"energy={a[2]}")
                print(f"det={a[3]}")
                print(f"log_det={a[4]}")
                loss_evolution.append(loss)
                loss_evolution_mismatch.append(a[1])
                loss_evolution_energy.append(a[2])
        except KeyboardInterrupt:
            break
    return [nn_f,loss_evolution,loss_evolution_mismatch,loss_evolution_energy,loss_evolution_growth]
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
        #X33=vrbl.repeat(int(X11.shape[0]/2),axis=0)
        # if id==0:
        #     #X3p=X33
        #     x3p=vrbl
        # else:
        #     X3p=np.vstack((X3p,X33))
        id+=1
        id2+=2
    id=0
    for i in range((n_mesh_x-1)*(n_mesh_y-1)):
        X3[id:id+2*(n_mesh_z-1),:,:]=X33_box
        id+=2*(n_mesh_z-1)
    return(X1,X2,X3)
with open('image_1_for_growth.pickle', 'rb') as f:
     S1 = pickle.load(f)
with open('image_2_for_growth.pickle', 'rb') as f:
     S2= pickle.load(f)
### program starts from here
x1_min_dom=-2*136/160
x1_max_dom=2*136/160
x2_min_dom=-2*192/160
x2_max_dom=2*192/160
x3_min_dom=-2
x3_max_dom=2
x1_min=0
x1_max_S1=S1.shape[2]
x1_max_S2=S2.shape[2]
x2_min=0
x2_max_S1=S1.shape[1]
x2_max_S2=S2.shape[1]
x3_min=0
x3_max_S1=S1.shape[0]
x3_max_S2=S2.shape[0]
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
muu=1
lam=1
time_steps=15
sudo_dt=1/time_steps
center=0
n_mesh_x=45
n_mesh_y=45
n_mesh_z=45
batch_size_x=4000
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
idn=np.arange(X1.shape[0])
idn=idn[0:np.shape(idn)[0]:2].tolist()
NN=init_params_b(layers,key)
start = time.time()
[final_data,loss_total_1,Loss_mismatch_1,Loss_energy_1,Loss_growth_1]=\
    training(NN,X1,X2,X3,batch_size_x,idn,.00005,50000)
end=time.time()
print(end-start)
loss_total=np.array(loss_total_1)[:,np.newaxis]; loss_mismatch=np.array(Loss_mismatch_1)[:,np.newaxis]
loss_energy=np.array(Loss_energy_1)[:,np.newaxis]; loss_growth=np.array(Loss_growth_1)[:,np.newaxis]
with open('NN_t_pre_fig7.pickle', 'wb') as f:
    pickle.dump(final_data, f)
with open('Total_Loss_pre_fig7.pickle', 'wb') as f:
    pickle.dump(loss_total, f)
with open('Mismatch_Loss_pre_fig7.pickle', 'wb') as f:
    pickle.dump(loss_mismatch, f)
with open('Energy_Loss_pre_fig7.pickle', 'wb') as f:
    pickle.dump(loss_energy, f)
