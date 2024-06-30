import pathlib
import matplotlib.pyplot as plt
import func_data_process as proc
import numpy as np
import os
# ######################################################################################
dir_data = 'data'
dir_data_mask = os.path.join(dir_data,'mask')

dir_results = 'results'
dir_results_iv = os.path.join(dir_results,'invivo')

# ######################################################################################
# subject used for test
print('load mask ...')
id_test =  [100,101,102,103,104,105,106,107,108,109,
            110,111,112,113,114,115,    117,118,119,
            120]
id_test_show = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]
rescale = [300.0,100.0]

model_names = [ 'unettv_{}_{}'.format(0.0,'exp_l2norm_sse'),
                'unettv_{}_{}'.format(0.0,'sqexp_l2norm_sse'),
                'unettv_{}_{}'.format(0.001,'sqexp_l2norm_sse'),
                # 'unettv_{}_{}'.format(0.003,'sqexp_l2norm_sse_tvp_l_0.1_new')]
                # 'unettv_{}_{}'.format(0.003,'sqexp_l2norm_sse_tvp_l_0.1_bs64')]
                # 'unettv_{}_{}'.format(0.003,'sqexp_l2norm_sse_tvp_l_0.1_bs128_step4')]
                'unettv_{}_{}'.format(0.003,'sqexp_l2norm_sse_tvp_l_0.1_bs32_a8p64s2_new')]
                
tes = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
methods = ['EXP','M$_2$NCM','M$_1$NCM','PCANR','UNet-EXP','UNet-M$_2$','UNet-TV','UNet-TVp']
# methods = ['EXP','M$_2$NCM','M$_1$NCM','PCANR','ResNet-EXP','ResNet-M$_2$','ResNet-TV','ResNet-TVp']

font_dir  = pathlib.Path(os.path.join('fonts','times.ttf'))
plt.rcParams['mathtext.fontset'] = "stix"
# --------------------------------------------------------------------------------------
# load mask
mask_body  = np.load(os.path.join(dir_data_mask,'mask_body.npy'))[id_test]
mask_liver = np.load(os.path.join(dir_data_mask,'mask_liver.npy'))[id_test]
mask_paren = np.load(os.path.join(dir_data_mask,'mask_parenchyma.npy'))[id_test]
print('> mask shape: ', mask_body.shape)

# --------------------------------------------------------------------------------------
# load results from conventional methods
# map_exp_iv   = np.load(os.path.join(dir_results_iv,'map_exp_new.npy'))[id_test]
# map_m1ncm_iv = np.load(os.path.join(dir_results_iv,'map_m1ncm_new.npy'))[id_test]
# map_m2ncm_iv = np.load(os.path.join(dir_results_iv,'map_m2ncm_new.npy'))[id_test]
# map_pcanr_iv = np.load(os.path.join(dir_results_iv,'map_pcanr_new.npy'))[id_test]

map_exp_iv   = np.load(os.path.join(dir_results_iv,'map_exp.npy'))[id_test]
map_m2ncm_iv = np.load(os.path.join(dir_results_iv,'map_m2ncm.npy'))[id_test]
map_m1ncm_iv = np.load(os.path.join(dir_results_iv,'map_m1ncm.npy'))[id_test]
map_pcanr_iv = np.load(os.path.join(dir_results_iv,'map_pcanr.npy'))[id_test]

# --------------------------------------------------------------------------------------
# load results from u-net
map_nets = []
for name in model_names:
    maps = np.load(os.path.join(dir_results_iv,name,'map.npy'))
    map_nets.append(maps)

# --------------------------------------------------------------------------------------
# load input data
imgs_iv = np.load(os.path.join(dir_results_iv,'imgs.npy'))

# ######################################################################################
# preprocessing
imgs_iv    = imgs_iv[id_test_show]*rescale[0]
mask_body  = mask_body[id_test_show]
mask_liver = mask_liver[id_test_show]
mask_paren = mask_paren[id_test_show]

map_all = []

map_all.append(map_exp_iv[id_test_show])
map_all.append(map_m2ncm_iv[id_test_show])
map_all.append(map_m1ncm_iv[id_test_show])
map_all.append(map_pcanr_iv[id_test_show])

for m in map_nets: 
    m_rs = m[id_test_show]*rescale
    m_rs_mask = proc.mask_out(imgs=m_rs,mask=mask_body)
    map_all.append(m_rs_mask)

imgs_iv = proc.mask_out(imgs=imgs_iv,mask=mask_body)

# ######################################################################################
# mean R2 in liver parenchyma
mean_all = []
for m in map_all:
    ave,_ = proc.mean_std_roi(m[...,1],   mask_paren)
    mean_all.append(ave)

# ######################################################################################
# Show maps
id_subject_show = [11,13]
# --------------------------------------------------------------------------------------
vmax_w  = 500
row,col = 3, 4
left_x, left_y,right_x,right_y = 2,9, 80,9
font_size = 8
# --------------------------------------------------------------------------------------
fig_width  = 7.16
fig_height = fig_width/8.0*2.8
# --------------------------------------------------------------------------------------
for i in id_subject_show:
    vmax_p  = mean_all[3][i]*1.5
    fig, axes = plt.subplots(nrows=row,ncols=col,figsize=(fig_width,fig_height),dpi=600,constrained_layout=True)
    [ax.set_axis_off() for ax in axes.ravel()]

    axes_weight = axes.ravel()[:4]
    axes_maps   = axes.ravel()[4:]

    # axes_weight[0].text(x=left_x,y=left_y,s='T$_2^*w$',color='white',font=font_dir,fontsize=font_size)
    for j,ax in enumerate(axes_weight):
        pcm_img = ax.imshow(imgs_iv[i,:,:,j],cmap='gray',interpolation='none',vmin=0,vmax=vmax_w)
        # ax.text(x=88,y=10,s='TE$_{}$={}'.format(j+1,tes[j]),color='white',font=font_dir,fontsize=font_size)

    for j,ax in enumerate(axes_maps):
        pcm_map=ax.imshow(map_all[j][i,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p)
        # ax.text(x=left_x,y=left_y,s=methods[j],color='white',font=font_dir,fontsize=font_size)
        # ax.text(x=93,y=10,s='R$_2^*$={:.0f}'.format(mean_all[j][i]),color='white',font=font_dir,fontsize=font_size)

    cbar1=fig.colorbar(pcm_img,ax=axes[0,:],shrink=0.7,aspect=24,ticks=[0,vmax_w/2,vmax_w])
    cbar2=fig.colorbar(pcm_map,ax=axes[1,:],shrink=0.7,aspect=24,ticks=[0,500,int(vmax_p)])
    cbar3=fig.colorbar(pcm_map,ax=axes[2,:],shrink=0.7,aspect=24,ticks=[0,500,int(vmax_p)])
    cbar1.ax.set_yticklabels(labels=[0,int(vmax_w/2),vmax_w],font=font_dir,fontsize=font_size)
    cbar2.ax.set_yticklabels(labels=[0,500,int(vmax_p)],font=font_dir,fontsize=font_size)
    cbar3.ax.set_yticklabels(labels=[0,500,int(vmax_p)],font=font_dir,fontsize=font_size)

    plt.savefig(os.path.join('figures','map_test_invivo_{}'.format(i)))

# ######################################################################################
##### Bland-Altman Analysis (In Vivo) ######
print('Bland-Altman plot analysing...')
xLimit,yLimit = 1100,100
row, col   = 2, 4
line_width = 1.0
font_size  = 7.5
# --------------------------------------------------------------------------------------
fig_width = 7.16
fig_height = fig_width/2.5
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=row,ncols=col,figsize=(fig_width,fig_height),dpi=600,constrained_layout=True)
axs = axes.ravel()
# --------------------------------------------------------------------------------------
ref_name = 'PCANR'
y   = mean_all[3]
axs[3].set_axis_off()
# --------------------------------------------------------------------------------------
for k in [0,1,2,4,5,6,7]:
    x = mean_all[k]
    mean = np.mean([x, y], axis=0)
    diff = x - y      # Difference between data1 and data2
    md   = np.mean(diff)        # Mean of the difference
    sd   = np.std(diff, axis=0) # Standard deviation of the difference

    axs[k].axhline(md, color='black', linestyle='-',linewidth = line_width)
    axs[k].axhline(md+1.96*sd, color='gray', linestyle='--',linewidth = line_width/2)
    axs[k].axhline(md-1.96*sd, color='gray', linestyle='--',linewidth = line_width/2)
    axs[k].plot(mean, diff,'o',color='blue',markersize=2.0) # data point

    if np.abs(md+1.96*sd) < yLimit:
        axs[k].text(x=40, y=md+1.96*sd+10,s='+1.96SD',font=font_dir,fontsize=font_size,color='black')
        axs[k].text(x=860,y=md+1.96*sd+20,s='{:>6.2f}'.format(md+1.96*sd),font=font_dir,fontsize=font_size,color='red')

    if np.abs(md-1.96*sd) < yLimit:
        axs[k].text(x=40, y=md-1.96*sd-20,s='-1.96SD',font=font_dir,fontsize=font_size,color='black')
        axs[k].text(x=860,y=md-1.96*sd-20,s='{:>6.2f}'.format(md-1.96*sd),font=font_dir,fontsize=font_size,color='red')

    axs[k].text(x=860,y=md+3,s='{:6.2f}'.format(md),font=font_dir,fontsize=font_size,color='red')

    axs[k].set_ylim([-yLimit,yLimit])
    axs[k].set_ylabel('{} - {} (s$^{{-1}}$)'.format(methods[k],ref_name),font=font_dir,fontsize=font_size)
    axs[k].set_xlim([0,xLimit])
    axs[k].set_xlabel('({} + {})/2 (s$^{{-1}}$)'.format(methods[k],ref_name),font=font_dir,fontsize=font_size)
    axs[k].set_xticks(ticks=[0,250,500,750,1000])
    axs[k].set_xticklabels(labels=[0,250,500,750,1000],font=font_dir)
    axs[k].set_yticks(ticks=[-100,-50,0,50,100])
    axs[k].set_yticklabels(labels=[-100,-50,0,50,100],font=font_dir)
    axs[k].tick_params(axis='both',direction='in',length=2.0,labelsize=7.5)

plt.savefig(os.path.join('figures','baplot_invivo.png'))

print(y)
# ######################################################################################
