# Calculate maps from in vivo data.
import os
import numpy as np
import matplotlib.pyplot as plt

import func_data_process as proc

###############################################################################
dir_data     = 'data'
dir_data_dcm = 'data_dcm'
dir_data_iv  = os.path.join('data','invivo')

reload_data = 1
remake_map  = 1

###############################################################################
# read image
print('='*98)
if os.path.exists(dir_data_iv) == False: os.mkdir(dir_data_iv)
dir_study = os.path.join(dir_data_dcm,'Study*')
info = proc.data_info(dir_study ,show=False)

print('Read Data...')
if reload_data == True:
    print('Reload in vivo data ...')
    imgs = proc.read_dicom_itk(dir_study,shape=(64,128,12))
    np.save(os.path.join(dir_data_iv,'img.npy'),imgs)

if reload_data == False:
    print('Load saved in vivo data ...')
    imgs = np.load(os.path.join(dir_data_iv,'img.npy'))

# load mask
mask_liver = np.load(os.path.join(dir_data,'mask_liver.npy'))
mask_body  = np.load(os.path.join(dir_data,'mask_body.npy'))
mask_paren = np.load(os.path.join(dir_data,'mask_parenchyma.npy'))

print('Data Info: ')
print('Data Shape: ',imgs.shape,mask_liver.shape,mask_body.shape,mask_paren.shape)

###############################################################################
# parameter mapping
print('='*98)
print('Parameter Mapping ...')
tes   = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])

if remake_map == True:
    print('Refit maps ...')
    maps_exp    = proc.image2map(imgs=imgs,tes=tes,method='PW',model='EXP',mask=mask_body,num_coils=1,fix_sigma=True)
    maps_m1ncm  = proc.image2map(imgs=imgs,tes=tes,method='PW',model='M1NCM',mask=mask_body,num_coils=1,fix_sigma=True)
    maps_m2ncm  = proc.image2map(imgs=imgs,tes=tes,method='PW',model='M2NCM',mask=mask_body,num_coils=1,fix_sigma=True)
    maps_pcanr  = proc.image2map(imgs=imgs,tes=tes,method='PCANR',model='M1NCM',mask=mask_body,num_coils=1,fix_sigma=True)
    maps_trunc  = proc.image2map(imgs=imgs,tes=tes,method='PW',model='Truncation',mask=mask_body,num_coils=1,fix_sigma=True)

    maps_exp    = proc.mask_out(maps_exp,mask_body)
    maps_m2ncm  = proc.mask_out(maps_m2ncm,mask_body)
    maps_m1ncm  = proc.mask_out(maps_m1ncm,mask_body)
    maps_pcanr  = proc.mask_out(maps_pcanr,mask_body)
    maps_trunc  = proc.mask_out(maps_trunc,mask_body)

    np.save(os.path.join(dir_data_iv,'map_exp.npy'),maps_exp)
    np.save(os.path.join(dir_data_iv,'map_m2ncm.npy'),maps_m2ncm)
    np.save(os.path.join(dir_data_iv,'map_m1ncm.npy'),maps_m1ncm)
    np.save(os.path.join(dir_data_iv,'map_pcanr.npy'),maps_pcanr)
    np.save(os.path.join(dir_data_iv,'map_trunc_nlm.npy'),maps_trunc)

if remake_map == False:
    print('Load existed maps ...')
    maps_exp    = np.load(os.path.join(dir_data_iv,'map_exp.npy'))
    maps_m2ncm  = np.load(os.path.join(dir_data_iv,'map_m2ncm.npy'))
    maps_m1ncm  = np.load(os.path.join(dir_data_iv,'map_m1ncm.npy'))
    maps_pcanr  = np.load(os.path.join(dir_data_iv,'map_pcanr.npy'))
    maps_trunc  = np.load(os.path.join(dir_data_iv,'map_trunc_nlm.npy'))

###############################################################################
# Show results
print('Show images, maps and masks ...')
sigma = proc.sigma_bkg(imgs,num_coil=1,mask=mask_body,mean=True)
N     = imgs.shape[0]
maps  = maps_pcanr
s_paren = proc.roi_average(imgs,mask_paren)
p_paren = proc.roi_average(maps,mask_paren)

# plt.figure(figsize=(44,22))
# for i in range(N):
#     plt.subplot(11,11,i+1),plt.imshow(imgs[i,...,0],cmap='gray',vmin=0.0,vmax=s_paren[i,0]*1.75)
#     plt.axis('off'),plt.colorbar(fraction=0.022)
#     plt.title('S = {:.1f}'.format(s_paren[i,1]),loc='left')
#     plt.title(label=i,loc='right'),plt.tight_layout()
# plt.savefig(os.path.join('figures','imgs'))
# print('Images done. ',end='')

# plt.figure(figsize=(44,22))
# for i in range(N):
#     plt.subplot(11,11,i+1),plt.imshow(maps[i,...,1],cmap='jet',vmin=0.0,vmax=r2_paren[i,1]*1.5),
#     plt.axis('off'),plt.colorbar(fraction=0.022)
#     plt.title('R2 = {:.1f}'.format(p_paren[i,1]),loc='left')
#     plt.title(label=i,loc='right'),plt.tight_layout()
# plt.savefig(os.path.join('figures','maps_pcanr_r2'))

plt.figure(figsize=(44,22))
for i in range(N):
    plt.subplot(11,11,i+1),plt.imshow(maps[i,...,0],cmap='gray',vmin=0.0,vmax=p_paren[i,0]*1.75),
    plt.axis('off'),plt.colorbar(fraction=0.022)
    plt.title('S0 = {:.1f}'.format(p_paren[i,0]),loc='left')
    plt.title(label=i,loc='right'),plt.tight_layout()
plt.savefig(os.path.join('figures','maps_pcanr_s0'))
print('Maps done.')

# mask_bkg = (mask_body+1)%2 # convert 1 to 0.
# plt.figure(figsize=(44,22))
# for i in range(N):
#     plt.subplot(11,11,i+1),plt.imshow(mask_bkg[i,...],cmap='gray'),
#     plt.axis('off'),plt.title(label=i,loc='right'),plt.title(label=np.round(sigma[i],decimals=2),loc='left')
# plt.savefig(os.path.join('figures','mask_bkg'))
# print('Masks done.')
print('='*98)