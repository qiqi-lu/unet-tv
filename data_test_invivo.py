# make training data
import os
import func_data_process as proc
import numpy as np
import config
import matplotlib.pyplot as plt

config.config_gpu(7)

###################################################################################################################
# Read in vivo data and map
dir_data = 'data'
# dir_data_iv   = 'data'
dir_data_iv   = os.path.join('data','invivo')
dir_data_simu = os.path.join('data','simu')
if os.path.exists(dir_data_simu) == False: os.mkdir(dir_data_simu)

###################################################################################################################
print('='*98)
print('Read Data...')
# Load image
imgs       = np.load(os.path.join(dir_data_iv,'img.npy'))

# Load mask
mask_liver = np.load(os.path.join(dir_data,'mask_liver.npy'))
mask_body  = np.load(os.path.join(dir_data,'mask_body.npy'))
mask_paren = np.load(os.path.join(dir_data,'mask_parenchyma.npy'))

# Load reference map
maps_refer = np.load(os.path.join(dir_data_iv,'map_pcanr.npy'))
maps_refer = proc.mask_out(maps_refer,mask_body)

tes   = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
sigma = proc.sigma_bkg(imgs,num_coil=1,mask=mask_body,mean=True)

print('Data Info: ')
print('Data Shape: ',imgs.shape)

###################################################################################################################
# Data for in vivo test
print('='*98)
print('Make in vivo test data ...')
id_test =  [100,101,102,103,104,105,106,107,108,109,
            110,111,112,113,114,115,    117,118,119,
            120]

imgs       = imgs[id_test]
maps_refer = maps_refer[id_test]
mask_paren = mask_paren[id_test]
mask_body  = mask_body[id_test]
sigma      = sigma[id_test]

print('-'*98)
print('Write to TFRecord file ...')
maps = maps_refer

imgs       = imgs.astype(np.float32)
maps       = maps.astype(np.float32)
tes        = np.repeat(np.reshape(tes,(1,-1)),repeats=imgs.shape[0],axis=0).astype(np.float32)
maps_sigma = np.ones_like(maps[...,0])[...,np.newaxis]*np.expand_dims(sigma,axis=(1,2,3))
maps_sigma = maps_sigma.astype(np.float32)
proc.write2TFRecord(patches_imgs=imgs,patches_maps=maps,tes=tes,sigma=maps_sigma,filename=os.path.join(dir_data_iv,'test_invivo'))

print('Dataset (test) Info:')
print('img shape:', imgs.shape)
print('map shape:', maps.shape)
print('tes shape:', tes.shape)
print('sigma shape:', maps_sigma.shape)

###################################################################################################################
# show test data images and maps 
s_paren = proc.roi_average(imgs,mask_paren)
plt.figure(figsize=(44,22))
for i in range(imgs.shape[0]):
    plt.subplot(11,11,i+1),plt.imshow(imgs[i,...,0],cmap='gray',vmin=0.0,vmax=s_paren[i,0]*1.5),plt.axis('off'),plt.colorbar(fraction=0.022)
    plt.title('S = {:.1f}'.format(s_paren[i,1]),loc='left')
    plt.title(label=i,loc='right')
plt.savefig(os.path.join('figures','imgs_test'))

r2_paren = proc.roi_average(maps,mask_paren)
plt.figure(figsize=(44,22))
for i in range(maps.shape[0]):
    plt.subplot(11,11,i+1),plt.imshow(maps[i,...,1],cmap='jet',vmin=0.0,vmax=r2_paren[i,1]*1.5),plt.axis('off'),plt.colorbar(fraction=0.022)
    plt.title('R2 = {:.1f}'.format(r2_paren[i,1]),loc='left')
    plt.title(label=i,loc='right')
plt.savefig(os.path.join('figures','maps_test_pcanr'))
print('='*98)