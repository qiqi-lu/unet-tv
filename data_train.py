# make training data
import os
import numpy as np
import tensorflow as tf

import func_data_process as proc
import config

config.config_gpu(5)
###############################################################################
dir_data_iv   = os.path.join('data','invivo')
dir_data_mask = os.path.join('data','mask')

###############################################################################
# Load image
print('Load saved in vivo data ...')
imgs = np.load(os.path.join(dir_data_iv,'img.npy'))
tes  = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])

# Load mask
mask_liver = np.load(os.path.join(dir_data_mask,'mask_liver.npy'))
mask_body  = np.load(os.path.join(dir_data_mask,'mask_body.npy'))
mask_paren = np.load(os.path.join(dir_data_mask,'mask_parenchyma.npy'))

sigma = proc.sigma_bkg(imgs,num_coil=1,mask=mask_body,mean=True)

print('Data Info: ')
print('Data Shape: ',imgs.shape,mask_liver.shape,mask_body.shape,mask_paren.shape)

###############################################################################
# Load reference maps
map_refer = maps_pcanr  = np.load(os.path.join(dir_data_iv,'map_pcanr.npy'))

###############################################################################
# make train data
print('='*98)
print('Make train data ...')
id_train = [0,  1,  2,      4,  5,  6,      8,
            10, 11, 12, 13,     15, 16, 17, 18, 19, 
            20,         23, 24, 25,         28, 29,
            30, 31, 32, 33, 34, 35, 36,         39,
                41, 42, 43, 44, 45,     47, 48,    
                    52, 53, 54, 55, 56, 57, 58, 59,
            60, 61,     63, 64, 65, 66, 67, 68, 69,
            70, 71,     73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 83, 84, 85, 86, 87,     89,
                91, 92, 93, 94, 95,     97, 98, 99,]

imgs_train       = imgs[id_train]
maps_refer_train = map_refer[id_train]
sigma_train      = sigma[id_train]
mask_paren_train = mask_paren[id_train]
mask_body_train  = mask_body[id_train]

N_train = imgs_train.shape[0]
print('Training dataset size: ',N_train)
print(sigma_train)

###############################################################################
print('Show training images and maps ...')
# s_paren_train = proc.roi_average(imgs_train,mask_paren_train)
# maps_train = maps_refer_train
# r2_paren_train = proc.roi_average(maps_train,mask_paren_train)

# plt.figure(figsize=(44,22))
# for i in range(N_train):
#     plt.subplot(11,11,i+1),plt.imshow(imgs_train[i,...,0],cmap='gray',vmin=0.0,vmax=s_paren_train[i,0]*1.5),plt.axis('off'),plt.colorbar(fraction=0.022)
#     plt.title('S = {:.1f}'.format(s_paren_train[i,1]),loc='left')
#     plt.title(label=i,loc='right')
#     plt.tight_layout()
# plt.savefig('figures/imgs_train')

# plt.figure(figsize=(44,22))
# for i in range(N_train):
#     plt.subplot(11,11,i+1),plt.imshow(maps_train[i,...,1],cmap='jet',vmin=0.0,vmax=r2_paren_train[i,1]*1.5),plt.axis('off'),plt.colorbar(fraction=0.022)
#     plt.title('R2 = {:.1f}'.format(r2_paren_train[i,1]),loc='left')
#     plt.title(label=i,loc='right')
#     plt.tight_layout()
# plt.savefig('figures/maps_train')

###############################################################################
print('='*98)
print('Patching ...')
# aug         = 6 # used
aug         = 8
# patch_size  = 32
patch_size  = 64
# step_size   = 8 # used
step_size   = 1

patches_imgs,idx = proc.patch(imgs_train,      mask=mask_body_train,patch_size=patch_size,step_size=step_size,aug=aug,id=True)
patches_maps     = proc.patch(maps_refer_train,mask=mask_body_train,patch_size=patch_size,step_size=step_size,aug=aug)
sigma            = proc.rearrange(sigma_train,idx=idx)

print('Write to TFRecord file ...')
patches_imgs    = patches_imgs.astype(np.float32)
patches_maps    = patches_maps.astype(np.float32)
tes             = np.repeat(np.reshape(tes,(1,-1)),repeats=patches_imgs.shape[0],axis=0).astype(np.float32)
maps_sigma      = np.ones_like(patches_maps[...,0])[...,np.newaxis]*np.expand_dims(sigma,axis=(1,2,3))
maps_sigma      = maps_sigma.astype(np.float32)
proc.write2TFRecord(patches_imgs=patches_imgs,patches_maps=patches_maps,tes=tes,sigma=maps_sigma,filename=os.path.join(dir_data_iv,'train_invivo_a8p64s1'))

print('Patch Info:')
print('imgs  shape:', patches_imgs.shape)
print('maps  shape:', patches_imgs.shape)
print('tes   shape:', tes.shape)
print('sigma shape:', sigma.shape)

###############################################################################
# Test saved TFRecords file.
print('='*98)
print('Read TFRecord file ...')
dataset_filenames = tf.io.gfile.glob(os.path.join(dir_data_iv,'train_invivo_a8p64s1.tfrecords'))
dataset           = tf.data.TFRecordDataset(dataset_filenames).map(proc.parse_all)
dataset_size      = proc.get_len(dataset)

print('Dataset:')
print(dataset_filenames)
print('dataset size: ',dataset_size)
for sample in dataset.batch(dataset_size).take(1):
    print('Patch shape:', sample[0].shape)
    print('Lable shape:', sample[1].shape)
    print('sigma shape:', sample[2].shape)

# for i in range(dataset_size):
#     plt.figure()
#     plt.subplot(2,2,1),plt.imshow(sample[0][i,...,0],cmap='gray'),plt.colorbar(fraction=0.022)
#     plt.subplot(2,2,2),plt.imshow(sample[2][i,...,0],cmap='gray'),plt.title(np.mean(sample[0][1][i,...,0]))
#     plt.subplot(2,2,3),plt.imshow(sample[1][i,...,0],cmap='gray'),plt.colorbar(fraction=0.022)
#     plt.subplot(2,2,4),plt.imshow(sample[1][i,...,1],cmap='jet'),plt.colorbar(fraction=0.022)
#     plt.savefig('figures/tmp')
print('='*98)

