import numpy as np
import os
import matplotlib.pyplot as plt

tv = np.linspace(0.01,1.0,100)
def tvp(tv,a,p):
    return np.where(tv<=a, 1.0/(p*a)*np.square(tv), a*np.log(tv)+1.0/p*a-a*np.log(a))

def tvp2(tv,a,p):
    return np.where(tv<=a, tv, a*np.log(tv)+2.0/p*a-a*np.log(a))

data_dir_iv = 'data'
map_m2ncm_iv = np.load(os.path.join(data_dir_iv,'map_m2ncm_new.npy'))
map_pcanr_iv = np.load(os.path.join(data_dir_iv,'map_pcanr_new.npy'))

def tv_map(x):
    eps  = 1e-6
    pad1 = np.array([[0,0],[0,1],[0,0],[0,0]])
    pad2 = np.array([[0,0],[0,0],[0,1],[0,0]])
    pixel_dif1 = x[:, 1:, :, :] - x[:, :-1, :, :] # finite forward difference donw->up
    pixel_dif2 = x[:, :, 1:, :] - x[:, :, :-1, :] # right->left
    pixel_dif1 = np.pad(pixel_dif1,pad1,"constant")
    pixel_dif2 = np.pad(pixel_dif2,pad2,"constant")
    tv = np.sqrt(np.square(pixel_dif1)+np.square(pixel_dif2)+np.square(eps))
    return tv

map_m2ncm_iv = map_m2ncm_iv/100.0
map_pcanr_iv = map_pcanr_iv/100.0

print(map_m2ncm_iv.shape)
tv_map_m2 = tv_map(map_m2ncm_iv)
tv_map_pc = tv_map(map_pcanr_iv)

plt.figure(figsize=(10,10),dpi=300)
plt.subplot(2,2,1)
plt.plot(tv,tvp(tv,0.2,2.0),label='square-log')
plt.plot(tv,tvp(tv,0.1,2.0),label='square-log')
plt.plot(tv,tvp(tv,0.05,2.0),label='square-log')
plt.plot(tv,tvp2(tv,0.2,2.0),label='linear-log')
plt.plot(tv,tvp2(tv,0.1,2.0),label='linear-log')
plt.plot(tv,tvp2(tv,0.15,2.0),label='linear-log')
plt.plot(tv,tv,color='black')
plt.legend()

idx = 111
plt.subplot(2,2,2)
plt.imshow(map_m2ncm_iv[idx,...,1],cmap='jet',vmax=11.0),plt.colorbar(fraction=0.022)

plt.subplot(2,2,3)
plt.imshow(tv_map_m2[idx,...,1],cmap='jet',vmax=4.0),plt.colorbar(fraction=0.022)

plt.subplot(2,2,4)
plt.imshow(tv_map_pc[idx,...,1],cmap='jet',vmax=0.2),plt.colorbar(fraction=0.022)

plt.savefig('figures/tmp')

