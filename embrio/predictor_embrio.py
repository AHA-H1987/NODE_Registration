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
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
def init_params_b(layers, key):
  Ws = []
  bs = []
  for i in range(len(layers) - 1):
    std_glorot = jnp.sqrt(2/(layers[i] + layers[i + 1]))
    key, subkey = jax.random.split(key)
    Ws.append(jax.random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    bs.append(jnp.zeros(layers[i + 1]))
  return (Ws, bs)
def init_params_b_zero(layers):
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
def Xs_reverse(x1x2,nn_f):
    X1X2=x1x2
    for i in range(time_steps):
        X1X2=X1X2-sudo_dt * forward_pass_b(X1X2,nn_f)
    return X1X2
@jit
def ODE_Euler_end_time(X1X2,nn_f):
    x1x2=X1X2
    for i in range(time_steps):
        x1x2_change=forward_pass_b(x1x2,nn_f)
        x1x2=x1x2+ sudo_dt * x1x2_change
    return x1x2
def F_values(X1X2,nn_f):
    dx1dX=vmap(grad(lambda x1x2: ODE_Euler_end_time(x1x2, nn_f)[0]))
    dx2dX = vmap(grad(lambda x1x2:ODE_Euler_end_time(x1x2, nn_f)[1]))
    F11_12=dx1dX(X1X2)
    F21_22=dx2dX(X1X2)
    return jnp.hstack((F11_12[:,[0]],F11_12[:,[1]],F21_22[:,[0]],F21_22[:,[1]]))
def predictor_and_gradiants(X1X2, F0,nn_f_p):
    F_values_p=F_values(X1X2, nn_f_p)
    F_values_p_material= jnp.hstack((F_values_p[:,[0]] * F0[:, [0]] + F_values_p[:,[1]]  * F0[:, [2]], F_values_p[:,[0]]  * F0[:, [1]] + F_values_p[:,[1]]  * F0[:, [3]],
                          F_values_p[:,[2]]  * F0[:, [0]] + F_values_p[:,[3]]  * F0[:, [2]], F_values_p[:,[2]]  * F0[:, [1]] + F_values_p[:,[3]]  * F0[:, [3]]))
    x1x2_p=ODE_Euler_end_time(X1X2,nn_f_p)
    return [ x1x2_p, F_values_p_material]
@jit
def dNN_dxs(x1x2,nn_f):
    dNN1_dxs_= vmap(grad(lambda x1x2_: forward_pass_b(x1x2_, nn_f)[0]))
    dNN2_dxs_= vmap(grad(lambda x1x2_: forward_pass_b(x1x2_, nn_f)[1]))
    return jnp.hstack((dNN1_dxs_(x1x2),dNN2_dxs_(x1x2)))

@jit
def xs_and_Fs(X1X2,Fps_values_mini,nn_f):
    x1x2=X1X2
    Fs_ =Fps_values_mini
    for i in range(time_steps):
        dNN_dxs_values = dNN_dxs(x1x2, nn_f)
        F11_p = 1 + sudo_dt * dNN_dxs_values[:,[0]]
        F12_p = sudo_dt* dNN_dxs_values[:,[1]]
        F21_p = sudo_dt* dNN_dxs_values[:,[2]]
        F22_p = 1 + sudo_dt * dNN_dxs_values[:,[3]]
        Fs_ = jnp.hstack((F11_p * Fs_[:, [0]] + F12_p * Fs_[:, [2]], F11_p * Fs_[:, [1]] + F12_p * Fs_[:, [3]],
                          F21_p * Fs_[:, [0]] + F22_p * Fs_[:, [2]], F21_p * Fs_[:, [1]] + F22_p * Fs_[:, [3]]))
        x1x2=x1x2+ sudo_dt * forward_pass_b(x1x2,nn_f)
    return x1x2,Fs_
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
def psi( Fs):
    F11 = Fs[:,[0]]
    F12 = Fs[:,[1]]
    F21 = Fs[:,[2]]
    F22 = Fs[:,[3]]
    C11 = F11 * F11 + F21 * F21
    C12 = F11 * F12 + F21 * F22
    C21 = C12
    C22 = F12 * F12 + F22 * F22
    trace_C = C11 + C22
    det_F = F11 * F22 - F12 * F21
    energy = (.5 * muu * (trace_C + 1 - 3) - muu * jnp.log(det_F) + .5 * lam * (jnp.log(det_F)) ** 2)
    return energy
@jit
def s1_s2(X1X2,x1x2,S1_mesh,S2_mesh):
    s1_X =s1(X1X2,S1_mesh)
    s2_phi_X =s2(x1x2,S2_mesh)
    ans = (s1_X - s2_phi_X) ** 2
    return ans
@jit
def integral(nn_f,X1X2,Fp_values_mini,S1_image,S2_image):
    x1x2,Fs=xs_and_Fs(X1X2,Fp_values_mini,nn_f)
    coef=jnp.ones_like(X1X2[:,[0]]) * dx * dy / 4
    var0=s1_s2(X1X2,x1x2,S1_image,S2_image)
    var1=psi(Fs)
    second_integrand=s1(X1X2,S1_image)* var1#S1_gauss
    first_integral=jnp.trace(jnp.matmul(jnp.transpose(var0), coef))
    second_integral=jnp.trace(jnp.matmul(jnp.transpose(second_integrand), coef))
    return [first_integral+second_integral/10, first_integral, second_integral,[],[]]
@jit
def loss_computation(nn_f, X1X2,Fp_values_mini,S1_image,S2_image):
    vrbl = integral(nn_f, X1X2,Fp_values_mini,S1_image,S2_image)
    loss0 = vrbl[0]
    return [loss0, vrbl[1], vrbl[2], vrbl[3], vrbl[4]]
def training(nn_f,X1,X2,batch_size_x,idn,S1_image,S2_image,lr, epoch_max:int=1000):
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
        Fp_values_mini = jnp.hstack((jnp.squeeze(Fp_values[0][intrand, :]).reshape((size, 1)), jnp.squeeze(Fp_values[1][intrand, :]).reshape((size, 1)), \
                                     jnp.squeeze(Fp_values[2][intrand, :]).reshape((size, 1)), jnp.squeeze(Fp_values[3][intrand, :]).reshape((size, 1))))
        try:
           a=loss_computation(nn_f,X1X2_mini,Fp_values_mini,S1_image,S2_image)
           loss=a[0]
           if jnp.isnan(loss):
               id_non=id_non+1
               print(id)
           #print(a[0])
           if ~(jnp.isnan(loss) | jnp.isinf(loss)):
               grad_for_opt=lambda nn_f: loss_computation(nn_f,X1X2_mini,Fp_values_mini,S1_image,S2_image)[0]
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
                loss_evolution.append(loss)
                loss_evolution_mismatch .append(a[1])
                loss_evolution_energy.append(a[2])
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
def img_extension(img,m):
    img=img[120:,40:450]
    #new_img=np.zeros(im_0[0:5,:].min(),im_0[0:5,:].max(),(img.shape[0]+2*m,img.shape[1]+2*m))
    new_img=np.zeros((img.shape[0]+2*m,img.shape[1]+2*m))
    new_img[m:m+img.shape[0],m:m+img.shape[1]]=img
    return new_img
def voxelization():
    x=np.arange(0,im_0.shape[1])
    y=np.arange(0,im_0.shape[0])
    x1,x2=np.meshgrid(x,y,indexing='xy')
    x1=x1.flatten()[:,None]
    x1=x1*(x1_max_dom-x1_min_dom)/(im_0.shape[1])+x1_min_dom
    x2=x2.flatten()[:,None]
    x2=x2*(x2_max_dom-x2_min_dom)/(im_0.shape[0])+x2_min_dom
    return np.hstack((x1,x2))
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

im_5=im_5/im_5.max()
im_4=im_4/im_4.max()
im_3=im_3/im_3.max()
# plt.imshow(im_3)
# plt.show()
im_2=im_2/im_2.max()
# plt.imshow(im_2)
# plt.show()
im_1=im_1/im_1.max()
# plt.imshow(im_1)
# plt.show()
im_0=im_0/im_0.max()
# plt.imshow(im_0)
# plt.show()
# plt.imshow(im_0)
im_0=img_extension(im_0,100)
im_1=img_extension(im_1,100)
im_2=img_extension(im_2,100)
im_3=img_extension(im_3,100)
im_4=img_extension(im_4,100)
im_5=img_extension(im_5,100)
S1_images=[im_0,im_1,im_2,im_3,im_4]
S2_images=[im_1,im_2,im_3,im_4,im_5]
S_inv=im_0
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
layers0=[2,5,5,5,2]
layers1=[2,40,40,40,2]
layers2=[2,40,40,40,2]
layers3=[2,40,40,40,2]
layers4=[2,40,40,40,2]
layers5=[2,40,40,40,2]
muu=1
lam=1
time_steps=15
sudo_dt=1/time_steps
n_points=10
center=0
n_mesh_x=200
n_mesh_y=200
batch_size_x=4000
x =np.linspace(x1_min_dom,x1_max_dom,n_mesh_x)
y =np.linspace(x2_min_dom,x2_max_dom,n_mesh_y)
dx=x[1]-x[0]
dy=y[1]-y[0]
X1,X2=weights_coordinates(x,y)
X1,X2=gausspoints(X1,X2)
X1,X2=column_coef(X1,X2,n_mesh_x,n_mesh_y)
X1=jnp.array(X1)
X2=jnp.array(X2)
size=X1.shape[0]*X2.shape[1]
idn=np.arange(X1.shape[0])
idn=idn[0:np.shape(idn)[0]:2].tolist()
NN_p0=init_params_b_zero(layers0)
NN_p1=init_params_b(layers1,key)
NN_p2=init_params_b(layers2,key)
NN_p3=init_params_b(layers3,key)
NN_p4=init_params_b(layers4,key)
NN_p5=init_params_b(layers5,key)
NN_ps=[NN_p1,NN_p2,NN_p3,NN_p4,NN_p5]
X1_whole = jnp.reshape(X1, (size, 1))
X2_whole = jnp.reshape(X2, (size, 1))
x1x2_i_1=jnp.hstack((X1_whole,X2_whole))
F_i_1= jnp.hstack((jnp.ones_like(X1_whole), jnp.zeros_like(X1_whole), jnp.zeros_like(X2_whole), jnp.ones_like(X2_whole)))
del X1_whole, X2_whole
NN_t_pres=[]
loss_total_list=[]
loss_energy_list=[]
loss_mismatch_list=[]
start = time.time()
for i in range(len(S2_images)):
    x1x2_p=x1x2_i_1
    Fp_values=F_i_1
    x1x2_p=np.hsplit(x1x2_p,2); Fp_values=np.hsplit(Fp_values,4)
    x1x2_p[0]=x1x2_p[0].reshape((X1.shape[0], X1.shape[1])); x1x2_p[1]=x1x2_p[1].reshape((X1.shape[0], X1.shape[1]))
    Fp_values[0]=Fp_values[0].reshape((X1.shape[0], X1.shape[1]));Fp_values[1]=Fp_values[1].reshape((X1.shape[0], X1.shape[1]))
    Fp_values[2]=Fp_values[2].reshape((X1.shape[0], X1.shape[1]));Fp_values[3]=Fp_values[3].reshape((X1.shape[0], X1.shape[1]))
    [final_data_1,loss_total_1,Loss_mismatch_1,Loss_energy_1,Loss_growth_1,id_non]=\
        training(NN_ps[i],X1,X2,batch_size_x,idn,S1_images[i],S2_images[i],.00005,50000)
    NN_t_pres.append(final_data_1)
    loss_total=np.array(loss_total_1)[:,np.newaxis]; loss_mismatch=np.array(Loss_mismatch_1)[:,np.newaxis]
    loss_energy=np.array(Loss_energy_1)[:,np.newaxis]; loss_growth=np.array(Loss_growth_1)[:,np.newaxis]
    loss_total_list.append(loss_total)
    loss_energy_list.append(loss_energy)
    loss_mismatch_list.append(loss_mismatch)
    if i!=int(len(S2_images)-1):
        x1x2_i_1,F_i_1=predictor_and_gradiants(x1x2_i_1,F_i_1, NN_t_pres[i])
end=time.time()
print(end-start)   
with open('NN_t_pre_fig6.pickle', 'wb') as f:
    pickle.dump(NN_t_pres, f)
with open('Total_Loss_pre_fig6.pickle', 'wb') as f:
    pickle.dump(loss_total_list, f)
with open('Mismatch_Loss_pre_fig6.pickle', 'wb') as f:
    pickle.dump(loss_mismatch_list, f)
with open('Energy_Loss_pre_fig6.pickle', 'wb') as f:
    pickle.dump(loss_energy_list, f)