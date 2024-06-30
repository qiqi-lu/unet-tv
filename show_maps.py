import numpy as np
import matplotlib.pyplot as plt
import os

dir_data    = 'data'
dir_data_iv = 'data'

id_test =  [100,101,102,103,104,105,106,107,108,109,
            110,111,112,113,114,115,    117,118,119,
            120]

mask_body  = np.load(os.path.join(dir_data,'mask_body.npy'))[id_test]
mask_liver = np.load(os.path.join(dir_data,'mask_liver.npy'))[id_test]
mask_paren = np.load(os.path.join(dir_data,'mask_parenchyma.npy'))[id_test]

map_exp_iv   = np.load(os.path.join(dir_data_iv,'map_exp.npy'))[id_test]
map_m1ncm_iv = np.load(os.path.join(dir_data_iv,'map_m1ncm.npy'))[id_test]
map_m2ncm_iv = np.load(os.path.join(dir_data_iv,'map_m2ncm.npy'))[id_test]
map_pcanr_iv = np.load(os.path.join(dir_data_iv,'map_pcanr.npy'))[id_test]
map_trunc_iv = np.load(os.path.join(dir_data_iv,'map_trunc_nlm.npy'))[id_test]

idx = 0
vm = 850.0
fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(12,4),dpi=300)
axes[0,0].imshow(map_exp_iv[idx,...,1],cmap='jet',vmin = 0.0,vmax =vm)
axes[0,1].imshow(map_m1ncm_iv[idx,...,1],cmap='jet',vmin = 0.0,vmax =vm)
axes[0,2].imshow(map_m2ncm_iv[idx,...,1],cmap='jet',vmin = 0.0,vmax =vm)
axes[1,0].imshow(map_pcanr_iv[idx,...,1],cmap='jet',vmin = 0.0,vmax =vm)
axes[1,1].imshow(map_trunc_iv[idx,...,1],cmap='jet',vmin = 0.0,vmax =vm)
plt.savefig('figures/tmp')

