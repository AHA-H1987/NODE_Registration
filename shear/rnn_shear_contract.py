import jax.numpy as jnp
from jax import random as rd
key = rd.PRNGKey(0)
import random
import pickle
import jax.nn as nn
import numpy as np
import time
import jax
from jax import grad, jit, vmap
from functools import partial
import optax
# from jax.scipy.optimize import minimize
# from jax.lax import scan
# from jax.nn import softplus
# import pickle
# import pandas as pd
from jax.config import config
config.update("jax_enable_x64", True)
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
def line(a,b):
    m=(b[[1],[0]]-a[[1],[0]])/(b[[0],[0]]-a[[0],[0]])
    b_p=-b[[0],[0]]*m+b[[1],[0]]
    return jnp.array([m,b_p])
def interpolation_constants(S_mesh,connect):
    size=connect.shape[0]
    line_constant=np.zeros((size,9))
    for i in range(size):
        if S_mesh[connect[i,2],0]-S_mesh[connect[i,1],0]!=0:
            line_constant[i,0]=(-S_mesh[connect[i,2],1]+S_mesh[connect[i,2],0]*(S_mesh[connect[i,2],1]-S_mesh[connect[i,1],1])/(S_mesh[connect[i,2],0]-S_mesh[connect[i,1],0]))/\
            (S_mesh[connect[i,0],1]-S_mesh[connect[i,2],1]-(S_mesh[connect[i,0],0]-S_mesh[connect[i,2],0])*(S_mesh[connect[i,2],1]-S_mesh[connect[i,1],1])/(S_mesh[connect[i,2],0]-S_mesh[connect[i,1],0]))

            line_constant[i, 1] = (- (S_mesh[connect[i, 2], 1] - S_mesh[connect[i, 1], 1]) / (S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 1], 0])) / \
                              (S_mesh[connect[i, 0], 1] - S_mesh[connect[i, 2], 1] - (S_mesh[connect[i, 0], 0] - S_mesh[connect[i, 2], 0]) * (S_mesh[connect[i, 2], 1] - S_mesh[connect[i, 1], 1]) / (S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 1], 0]))
            line_constant[i, 2] = (1) / \
                              (S_mesh[connect[i, 0], 1] - S_mesh[connect[i, 2], 1] - (S_mesh[connect[i, 0], 0] - S_mesh[connect[i, 2], 0]) * (S_mesh[connect[i, 2], 1] - S_mesh[connect[i, 1], 1]) / ( S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 1], 0]))
        else:
            line_constant[i, 0] =-S_mesh[connect[i,2],0]/(S_mesh[connect[i,0],0]-S_mesh[connect[i,2],0])
            line_constant[i, 1] = 1 / (S_mesh[connect[i, 0], 0] - S_mesh[connect[i, 2], 0])
            line_constant[i, 2] =0
        if S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 0], 0]!= 0:
                line_constant[i, 3] = (-S_mesh[connect[i, 2], 1] + S_mesh[connect[i, 2], 0] * (S_mesh[connect[i, 2], 1] - S_mesh[connect[i, 0], 1]) / (S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 0], 0])) / \
                              (S_mesh[connect[i, 1], 1] - S_mesh[connect[i, 2], 1] - (S_mesh[connect[i, 1], 0] - S_mesh[connect[i, 2], 0]) * (S_mesh[connect[i, 2], 1] - S_mesh[connect[i, 0], 1]) / (S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 0], 0]))
                line_constant[i, 4] = (- (S_mesh[connect[i, 2], 1] - S_mesh[connect[i, 0], 1]) / (S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 0], 0])) / \
                              (S_mesh[connect[i, 1], 1] - S_mesh[connect[i, 2], 1] - (S_mesh[connect[i, 1], 0] - S_mesh[connect[i, 2], 0]) * (S_mesh[connect[i, 2], 1] - S_mesh[connect[i, 0], 1]) / (S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 0], 0]))
                line_constant[i,5]  = (1) / \
                             (S_mesh[connect[i, 1], 1] - S_mesh[connect[i, 2], 1] - ( S_mesh[connect[i, 1], 0] - S_mesh[connect[i, 2], 0]) * (S_mesh[connect[i, 2], 1] - S_mesh[connect[i, 0], 1]) / (S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 0], 0]))
        else:
            line_constant[i, 3] = -S_mesh[connect[i, 2], 0] / (S_mesh[connect[i, 1], 0] - S_mesh[connect[i, 2], 0])
            line_constant[i, 4] = 1 / (S_mesh[connect[i, 1], 0] - S_mesh[connect[i, 2], 0])
            line_constant[i, 5] = 0
        if S_mesh[connect[i, 1], 0] - S_mesh[connect[i, 0], 0] != 0:
                line_constant[i, 6] = (-S_mesh[connect[i, 1], 1] + S_mesh[connect[i, 1], 0] * (S_mesh[connect[i, 1], 1] - S_mesh[connect[i, 0], 1]) / (S_mesh[connect[i, 1], 0] - S_mesh[connect[i, 0], 0])) / \
                              (S_mesh[connect[i, 2], 1] - S_mesh[connect[i, 1], 1] - (S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 1], 0]) * (S_mesh[connect[i, 1], 1] - S_mesh[connect[i, 0], 1]) / (S_mesh[connect[i, 1], 0] - S_mesh[connect[i, 0], 0]))
                line_constant[i, 7] = (- (S_mesh[connect[i, 1], 1] - S_mesh[connect[i, 0], 1]) / (S_mesh[connect[i, 1], 0] - S_mesh[connect[i, 0], 0])) / \
                              (S_mesh[connect[i, 2], 1] - S_mesh[connect[i, 1], 1] - (S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 1], 0]) * (S_mesh[connect[i, 1], 1] - S_mesh[connect[i, 0], 1]) / (S_mesh[connect[i, 1], 0] - S_mesh[connect[i, 0], 0]))

                line_constant[i,8]  = (1) / \
                              (S_mesh[connect[i, 2], 1] - S_mesh[connect[i, 1], 1] - (S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 1], 0]) * (S_mesh[connect[i, 1], 1] - S_mesh[connect[i, 0], 1]) / (S_mesh[connect[i, 1], 0] - S_mesh[connect[i, 0], 0]))
        else:
            line_constant[i, 6] = -S_mesh[connect[i, 1], 0] / (S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 1], 0])
            line_constant[i, 7] = 1 / (S_mesh[connect[i, 2], 0] - S_mesh[connect[i, 1], 0])
            line_constant[i, 8] = 0
    return line_constant
def S1S2_values(x1x2, constants,S_mesh,connect):
    S_mesh_values = S_mesh[:, [2]]
    side_1_values = constants[:, [0]] + jnp.matmul(constants[:, [1]],jnp.transpose(x1x2[:,[0]]))+ jnp.matmul(constants[:, [2]],jnp.transpose(x1x2[:,[1]]))
    side_2_values = constants[:, [3]] + jnp.matmul(constants[:, [4]],jnp.transpose(x1x2[:,[0]])) + jnp.matmul(constants[:, [5]],jnp.transpose(x1x2[:,[1]]))
    side_3_values = constants[:, [6]] + jnp.matmul(constants[:, [7]],jnp.transpose(x1x2[:,[0]])) + jnp.matmul(constants[:, [8]],jnp.transpose(x1x2[:,[1]]))
    element_searching_bolean = jnp.where((side_1_values >= 0) & (side_2_values >= 0) & (side_3_values >= 0),
                                        jnp.ones_like(side_1_values), jnp.zeros_like(side_1_values))
    N_element=jnp.sum(element_searching_bolean ,axis=0)
    N_interpolation=jnp.where(N_element==0,jnp.transpose(jnp.zeros_like(x1x2[:, [0]])),1/N_element)
    S =side_1_values  * S_mesh_values[connect[:, 0]] + side_2_values  * S_mesh_values[
        connect[:, 1]] + \
                side_3_values * S_mesh_values[connect[:, 2]]
    S=jnp.transpose(jnp.sum(S*element_searching_bolean,axis=0)*N_interpolation)
    return S
@jit
def x_value(X1X2,nn_f):
    return X1X2+ forward_pass_b(X1X2,nn_f)
@jit
def F_values(X1,X2,nn_f):
    dx1dX=vmap(grad(lambda X1X2_: x_value(X1X2_,nn_f)[0]))
    dx2dX = vmap(grad(lambda X1X2_:x_value(X1X2_,nn_f)[1]))
    F11_12=dx1dX(jnp.hstack((X1,X2)))
    F21_22=dx2dX(jnp.hstack((X1,X2)))
    return jnp.hstack((F11_12[:,[0]],F11_12[:,[1]],F21_22[:,[0]],F21_22[:,[1]]))
#F_values_batched=vmap(F_values,in_axes=)
@jit
def psi(Fs):
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
    # trace_C=jnp.where(det_F>0,trace_C,2*jnp.ones_like(det_F))
    # det_F=jnp.where(det_F>0,det_F,jnp.ones_like(det_F))
    energy = (.5 * muu * (trace_C + 1 - 3) - muu * jnp.log(det_F) + .5 * lam * (jnp.log(det_F)) ** 2)
    return energy
@jit
def s1_s2(S1_gauss,x1x2):
    s1_X = S1_gauss
    s2_phi_X = S1S2_values(x1x2, constants_S2, S2_mesh, connect)
    ans = (s1_X - s2_phi_X) ** 2
    return ans
@jit
def integral(nn_f,X1,X2,S1_gauss):
    coef=jnp.ones_like(X1) * dx * dy / 4
    x_s=x_value(jnp.hstack((X1,X2)),nn_f)
    F_s=F_values(X1,X2,nn_f)
    var0 = s1_s2(S1_gauss, x_s)
    var1=psi(F_s)
    #var1 = gradient(nn_f, X1, X2)
    #var2=S1_gauss*var2
    second_integrand = S1_gauss* var1
    #third_integrand = second_integrand
    first_integral=jnp.trace(jnp.matmul(jnp.transpose(var0), coef))
    second_integral = jnp.trace(jnp.matmul(jnp.transpose(second_integrand), coef))
    #var2 = jnp.trace(jnp.matmul(jnp.transpose(var2), coef))
    # + second_integral
    #+ second_integral / 1
    return [(9000*first_integral + second_integral), first_integral, second_integral,second_integral,[]]
@jit
def loss_computation(nn_f,  X1, X2,S1_gauss):
    vrbl = integral(nn_f, X1, X2,S1_gauss)
    loss0 = vrbl[0]
    # loss1=torch.zeros(1,1)
    return [loss0, vrbl[1], vrbl[2], vrbl[3], vrbl[4]]
loss_computation_batched=vmap(loss_computation,(None,0,0))
def training(nn_f,X1,X2,batch_size_x,idn,lr, epoch_max:int=1000):
    loss_evolution = []
    loss_evolution_mismatch = []
    loss_evolution_energy = []
    loss_evolution_det = []
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(nn_f)
    #optimizer=torch.optim.Adam(nn_f.parameters(), lr=)
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
        S1_mini = jnp.squeeze(S1[intrand, :])
        size = X1_mini.shape[0] * X1_mini.shape[1]
        X1_mini = jnp.reshape(X1_mini, (size, 1))
        X2_mini = jnp.reshape(X2_mini, (size, 1))
        S1_mini = jnp.reshape(S1_mini, (size, 1))
        # start = time.time()
        # forward_Euler = nn_f.forward(X1_mini, X2_mini)
        # end=time.time()
        # print(end-start)
        try:
           #start = time.time()
           a=loss_computation(nn_f,X1_mini, X2_mini,S1_mini)
           loss=a[0]
           if ~(jnp.isnan(loss) | jnp.isinf(loss)):
               grad_for_opt=lambda nn_f: loss_computation(nn_f,X1_mini,X2_mini,S1_mini)[0]
               grads = jax.grad(grad_for_opt)(nn_f)
               updates, opt_state = optimizer.update(grads, opt_state)
               nn_f = optax.apply_updates(nn_f, updates)
           #end = time.time()
           #print(end-start)
           if i % 10==0:
                print(f'epoch={i}, loss={loss}')
                print(f"imag_mismatch={a[1]}")
                print(f"det={a[2]}")
                print(f"energy={a[3]}")
                print(f"log_det={a[4]}")
                loss_evolution.append(loss)
                loss_evolution_mismatch .append(a[1])
                loss_evolution_det.append(a[2])
                loss_evolution_energy.append(a[3])
                # end = time.time()
                # print(end-start)
                # start = time.time()
        except KeyboardInterrupt:
            break
    return [nn_f,loss_evolution,loss_evolution_mismatch,loss_evolution_energy, loss_evolution_energy]
def img_extension(img):
    sz = np.shape(img)
    zero1 = jnp.zeros((int(sz[0]), int(sz[1] / 2)))
    zero2 = jnp.zeros((int(sz[0] / 2), int(2 * sz[1])))
    img = jnp.concatenate((img, zero1), axis=1)
    img= jnp.concatenate((zero1, img), axis=1)
    img= jnp.concatenate((zero2, img), axis=0)
    img= jnp.concatenate((img, zero2), axis=0)
    return img
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
with open("Plate_shear_connectivity.txt") as connectivity:
    ad=0
    for i in connectivity:
        if ad==0:
            j = i.split()
            connect= [[int(e) for e in j]]
        else:
            j = i.split()
            num=[[int(e) for e in j]]
            connect= np.append(connect, num, axis=0)
        ad=1
with open("Plate_shear_mesh_S1.txt") as S1:
    ad=0
    for i in S1:
        if ad==0:
            j = i.split()
            S1_mesh= [[float(e) for e in j]]
        else:
            j = i.split()
            num=[[float(e) for e in j]]
            S1_mesh= np.append(S1_mesh, num, axis=0)
        ad=1
with open("Plate_shear_mesh_S2.txt") as S2:
    ad=0
    for i in S2:
        if ad==0:
            j = i.split()
            S2_mesh= [[float(e) for e in j]]
        else:
            j = i.split()
            num=[[float(e) for e in j]]
            S2_mesh= np.append(S2_mesh, num, axis=0)
        ad=1
### program starts from here
constants_S2 = interpolation_constants(S2_mesh, connect)
layers=[2,40,40,40,2]
muu=1
lam=1
time_steps=15
sudo_dt=1/time_steps
n_points=10
center=0
n_mesh_x=200
n_mesh_y=200
batch_size_x=4000
x =np.linspace(-2,2,n_mesh_x)
y =np.linspace(-2,2,n_mesh_y)
dx=x[1]-x[0]
dy=y[1]-y[0]
X1,X2=weights_coordinates(x,y)
X1,X2=gausspoints(X1,X2)
X1,X2=column_coef(X1,X2,n_mesh_x,n_mesh_y)
X1=jnp.array(X1)
X2=jnp.array(X2)
size=X1.shape[0]*X2.shape[1]
X1X2_whole=jnp.hstack((X1.reshape((size,1)),X2.reshape((size,1))))
constants_S1= interpolation_constants(S1_mesh, connect)
S1=S1S2_values(X1X2_whole, constants_S1,S1_mesh,connect)
S1=S1.reshape((X1.shape[0],X2.shape[1]))
X1=X1-center
X2=X2-center
idn=np.arange(X1.shape[0])
idn=idn[0:np.shape(idn)[0]:2].tolist()
NN=init_params_b(layers,key)
start = time.time()
[final_data,loss_total_1,Loss_mismatch_1,Loss_det_1,Loss_energy_1]=\
    training(NN,X1,X2,batch_size_x,idn,.00005,100000)
end=time.time()
print(end-start)
loss_total=np.array(loss_total_1)[:,np.newaxis]; loss_mismatch=np.array(Loss_mismatch_1)[:,np.newaxis]
loss_det=np.array(Loss_det_1)[:,np.newaxis]; loss_energy=np.array(Loss_energy_1)[:,np.newaxis]
with open('RNN_trained.pickle', 'wb') as f:
    pickle.dump(final_data, f)
jnp.save('Total_Loss_RNN.npy', np.array(loss_total, dtype=object), allow_pickle=True)
jnp.save('Mismatch_Loss_RNN.npy', np.array(loss_mismatch, dtype=object), allow_pickle=True)
jnp.save('Energy_Loss_RNN.npy', np.array(loss_energy, dtype=object), allow_pickle=True)
jnp.save('Det_Loss_RNN.npy', np.array(loss_det, dtype=object), allow_pickle=True)