import numpy as np
import matplotlib.pyplot as plt
import func_data_process as proc
import metricx
import os
# ######################################################################################
dir_data = 'data'
dir_mask = os.path.join(dir_data,'mask')

dir_results = 'results'
dir_results_simu = os.path.join(dir_results,'simu')
# --------------------------------------------------------------------------------------
id_test =  [100,101,102,103,104,105,106,107,108,109,
            110,111,112,113,114,115,    117,118,119,
            120]
id_test_show = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,18,19]
rescale      = [300.0,100.0]
sigma_simu   = 7.5

model_names = [ 'unettv_{}_{}'.format(0.0,'sqexp_l2norm_sse'),
                'unettv_{}_{}'.format(0.002,'sqexp_l2norm_sse_tvp_l_0.2_new'),
                'unettv_{}_{}'.format(0.003,'sqexp_l2norm_sse_tvp_l_0.1_bs32_a8p64s2_new'),
                'unettv_{}_{}'.format(0.004,'sqexp_l2norm_sse_tvp_l_0.1'),
                ]
methods = ['$\lambda$ = 0','$\lambda$ = 0.002','$\lambda$ = 0.003','$\lambda$ = 0.004',]

# ######################################################################################
# load results
# load mask
mask_body  = np.load(os.path.join(dir_mask,'mask_body.npy'))[id_test]
mask_liver = np.load(os.path.join(dir_mask,'mask_liver.npy'))[id_test]
mask_paren = np.load(os.path.join(dir_mask,'mask_parenchyma.npy'))[id_test]
print('> mask shape: ', mask_body.shape)
# --------------------------------------------------------------------------------------
# load results from u-nets
map_nets = []
for name in model_names:
    maps = np.load(os.path.join(dir_results_simu,name,'sigma_'+str(sigma_simu),'map.npy'))
    map_nets.append(maps)
# --------------------------------------------------------------------------------------
map_gt_simu = np.load(os.path.join(dir_results_simu,'map_simu_gt.npy'.format(sigma_simu)))

# ######################################################################################
# preprocessing
map_gt_simu = map_gt_simu[id_test_show]*rescale
map_gt_simu = proc.mask_out(map_gt_simu,mask_body)

map_all = []
for m in map_nets:
    m_rs = m[id_test_show]*rescale
    m_rs_mask = proc.mask_out(m_rs,mask_body)
    map_all.append(m_rs_mask)

N_methods = len(map_all)

# ######################################################################################
# NRMSE & SSIM Evaluation
print('='*98)
print('NRMSE and SSIM Evaluating ...')
nrmses, ssims = [], []
for mp in map_all:
    e = metricx.nRMSE(map_gt_simu[...,1],mp[...,1],mask=mask_liver,mean=False)
    s = metricx.SSIM(map_gt_simu[...,1],mp[...,1], mask=mask_liver,mean=False)
    nrmses.append(e)
    ssims.append(s)

# ######################################################################################
# Mean R2* in ROI
mean_gt,_ = proc.mean_std_roi(map_gt_simu[...,1],mask_paren)
mean_all  = []
for mp in map_all:
    m,_ = proc.mean_std_roi(mp[...,1],mask_paren)
    mean_all.append(m)

# ######################################################################################
# parameter maps
i_sub = 13
print('NRMSE')
for i in range(4):
    print(nrmses[i][i_sub])
print('SSIM')
for i in range(4):
    print(ssims[i][i_sub])
# --------------------------------------------------------------------------------------
vmax_p      = mean_gt[i_sub]*1.5
vmax_p_diff = mean_gt[i_sub]*0.5
row, col    = 2, 5
font_size   = 8
left_x, left_y, right_x, right_y = 2, 9, 80, 9
# --------------------------------------------------------------------------------------
fig_width  = 7.16
fig_height = fig_width/3.75
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=row,ncols=col,figsize=(fig_width,fig_height),dpi=600,constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]

lc, rc = 15, -10
# --------------------------------------------------------------------------------------
axes[0,0].imshow(map_gt_simu[i_sub,:,lc:rc,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p)
axes[0,0].set_title('GT',fontsize=font_size)
# axes[0].text(x=right_x+30,y=right_y,s='R$_2^*$',color='white',fontsize=font_size)
# axes[0].text(x=left_x,y=left_y,s='GT',color='white',fontsize=font_size)
# --------------------------------------------------------------------------------------
# Reconstructed maps
for i_method in range(N_methods):
    pcm_map = axes[0,i_method+1].imshow(map_all[i_method][i_sub,:,lc:rc,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p)
    axes[0,i_method+1].set_title(methods[i_method],fontsize=font_size)
    axes[1,i_method+1].imshow(np.abs(map_all[i_method][i_sub,:,lc:rc,1]-map_gt_simu[i_sub,:,lc:rc,1]),cmap='jet',interpolation='none',vmin=0,vmax=vmax_p/3)
    # axes[i_method+1].text(x=left_x,y=left_y,s=methods[i_method],color='white',fontsize=font_size)
    # axes[i_method+1].text(x=right_x+10,y=right_y,s='{:.4f}'.format(ssims[i_method][i_sub]),color='white',fontsize=font_size)
# --------------------------------------------------------------------------------------
cb = fig.colorbar(pcm_map,ax=axes[0,:],shrink=0.7,aspect=24,ticks=[0,500,vmax_p])
cb.ax.tick_params(labelsize=font_size)
plt.savefig(os.path.join('figures','map_test_simu_{}_hyperparam'.format(i_sub)))
