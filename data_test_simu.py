# make training data
import os
import func_data_process as proc
import numpy as np
import config
import matplotlib.pyplot as plt

config.config_gpu(7)

###################################################################################################################
# Read in vivo data and map
dir_data      = 'data'
# dir_data_iv   = 'data'
dir_data_iv   = os.path.join('data','invivo')
dir_data_simu = os.path.join('data','simu')
if os.path.exists(dir_data_simu) == False: os.mkdir(dir_data_simu)

remake = True

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
# Choose data for test
id_test =  [100,101,102,103,104,105,106,107,108,109,
            110,111,112,113,114,115,    117,118,119,
            120]

imgs       = imgs[id_test]
maps_refer = maps_refer[id_test]
mask_paren = mask_paren[id_test]
mask_body  = mask_body[id_test]
sigma      = sigma[id_test]

print('Data Shape (test): ',imgs.shape)

###################################################################################################################
print('='*98)
print('Make simulated test data ...')

maps_simu_gt = np.copy(maps_refer)
sigma_simu   = 7.5
tes_simu     = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
size         = maps_simu_gt.shape[0]

# mapping simulated noisy images to maps
if remake:
    imgs_simu_gt = proc.map2image(maps=maps_simu_gt,tes=tes_simu,map_type='R2')
    imgs_simu_n  = proc.addNoise(imgs=imgs_simu_gt,sigma=sigma_simu,noise_type='Rician',NCoils=1,random_seed=8200)

    maps_simu_exp   = proc.image2map(imgs=imgs_simu_n,tes=tes_simu,method='PW',model='EXP',mask=mask_body,num_coils=1,fix_sigma=True)
    maps_simu_m1ncm = proc.image2map(imgs=imgs_simu_n,tes=tes_simu,method='PW',model='M1NCM',mask=mask_body,num_coils=1,fix_sigma=True)
    maps_simu_m2ncm = proc.image2map(imgs=imgs_simu_n,tes=tes_simu,method='PW',model='M2NCM',mask=mask_body,num_coils=1,fix_sigma=True)
    maps_simu_pcanr = proc.image2map(imgs=imgs_simu_n,tes=tes_simu,method='PCANR',model='M1NCM',mask=mask_body,num_coils=1,fix_sigma=True)


    maps_simu_exp   = proc.mask_out(maps_simu_exp,mask_body)
    maps_simu_m1ncm = proc.mask_out(maps_simu_m1ncm,mask_body)
    maps_simu_m2ncm = proc.mask_out(maps_simu_m2ncm,mask_body)
    maps_simu_pcanr = proc.mask_out(maps_simu_pcanr,mask_body)

    np.save(os.path.join(dir_data_simu,'imgs_simu_gt.npy'),imgs_simu_gt)
    np.save(os.path.join(dir_data_simu,'imgs_simu_n_{}.npy'.format(sigma_simu)),   imgs_simu_n)
    np.save(os.path.join(dir_data_simu,'map_simu_exp_{}.npy'.format(sigma_simu)),  maps_simu_exp)
    np.save(os.path.join(dir_data_simu,'map_simu_m1ncm_{}.npy'.format(sigma_simu)),maps_simu_m1ncm)
    np.save(os.path.join(dir_data_simu,'map_simu_m2ncm_{}.npy'.format(sigma_simu)),maps_simu_m2ncm)
    np.save(os.path.join(dir_data_simu,'map_simu_pcanr_{}.npy'.format(sigma_simu)),maps_simu_pcanr)
else:
    imgs_simu_gt    = np.load(os.path.join(dir_data_simu,'imgs_simu_gt.npy'.format(sigma_simu)))
    imgs_simu_n     = np.load(os.path.join(dir_data_simu,'imgs_simu_n_{}.npy'.format(sigma_simu)))
    maps_simu_exp   = np.load(os.path.join(dir_data_simu,'map_simu_exp_{}.npy'.format(sigma_simu)))
    maps_simu_m1ncm = np.load(os.path.join(dir_data_simu,'map_simu_m1ncm_{}.npy'.format(sigma_simu)))
    maps_simu_m2ncm = np.load(os.path.join(dir_data_simu,'map_simu_m2ncm_{}.npy'.format(sigma_simu)))
    maps_simu_pcanr = np.load(os.path.join(dir_data_simu,'map_simu_pcanr_{}.npy'.format(sigma_simu)))

###################################################################################################################
# Save to TFRecord file.
imgs_simu_n     = imgs_simu_n.astype(np.float32)
maps_simu_gt    = maps_simu_gt.astype(np.float32)
tes_simu        = np.repeat(np.reshape(tes_simu,(1,-1)),repeats=size,axis=0).astype(np.float32)
maps_sigma_simu = np.ones_like(maps_simu_gt[...,0])[...,np.newaxis]*np.expand_dims(np.repeat(sigma_simu,repeats=size),axis=(1,2,3))
maps_sigma_simu = maps_sigma_simu.astype(np.float32)
proc.write2TFRecord(patches_imgs=imgs_simu_n,patches_maps=maps_simu_gt,tes=tes_simu,sigma=maps_sigma_simu,filename=os.path.join(dir_data_simu,'test_simu_{}'.format(sigma_simu)))

print('Dataset:')
print('Imgae shape:', imgs_simu_n.shape)
print('Label shape:', maps_simu_gt.shape)
print('Sigma shape:', maps_sigma_simu.shape)

###################################################################################################################
# Show result.
k=0
plt.figure(figsize=(36,6),dpi=300)
for i in range(12):
    plt.subplot(3,12,i+1),plt.imshow(imgs_simu_gt[k,...,i],cmap='gray',vmin=0.0,vmax=500.0),plt.colorbar(fraction=0.022),plt.axis('off')
    plt.subplot(3,12,i+1+12),plt.imshow(imgs_simu_n[k,...,i],cmap='gray',vmin=0.0,vmax=500.0),plt.colorbar(fraction=0.022),plt.axis('off')
plt.subplot(3,12,25),plt.imshow(maps_simu_gt[k,...,0],   cmap='gray',vmin=0.0,vmax=500.0),plt.colorbar(fraction=0.022),plt.axis('off'),plt.title('GT')
plt.subplot(3,12,26),plt.imshow(maps_simu_gt[k,...,1],   cmap='jet',vmin=0.0,vmax=1000.0),plt.colorbar(fraction=0.022),plt.axis('off')
plt.subplot(3,12,27),plt.imshow(maps_simu_exp[k,...,0],  cmap='gray',vmin=0.0,vmax=500.0),plt.colorbar(fraction=0.022),plt.axis('off'),plt.title('EXP')
plt.subplot(3,12,28),plt.imshow(maps_simu_exp[k,...,1],  cmap='jet',vmin=0.0,vmax=1000.0),plt.colorbar(fraction=0.022),plt.axis('off')
plt.subplot(3,12,29),plt.imshow(maps_simu_m2ncm[k,...,0],cmap='gray',vmin=0.0,vmax=500.0),plt.colorbar(fraction=0.022),plt.axis('off'),plt.title('M2NCM')
plt.subplot(3,12,30),plt.imshow(maps_simu_m2ncm[k,...,1],cmap='jet',vmin=0.0,vmax=1000.0),plt.colorbar(fraction=0.022),plt.axis('off')
plt.subplot(3,12,31),plt.imshow(maps_simu_m1ncm[k,...,0],cmap='gray',vmin=0.0,vmax=500.0),plt.colorbar(fraction=0.022),plt.axis('off'),plt.title('M1NCM')
plt.subplot(3,12,32),plt.imshow(maps_simu_m1ncm[k,...,1],cmap='jet',vmin=0.0,vmax=1000.0),plt.colorbar(fraction=0.022),plt.axis('off')
plt.subplot(3,12,33),plt.imshow(maps_simu_pcanr[k,...,0],cmap='gray',vmin=0.0,vmax=500.0),plt.colorbar(fraction=0.022),plt.axis('off'),plt.title('PCANR')
plt.subplot(3,12,34),plt.imshow(maps_simu_pcanr[k,...,1],cmap='jet',vmin=0.0,vmax=1000.0),plt.colorbar(fraction=0.022),plt.axis('off')
plt.savefig('figures/image_test_simu_{}'.format(k))

print('='*98)
