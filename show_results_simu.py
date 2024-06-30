import pathlib
import numpy as np
import os
import pandas
import matplotlib.pyplot as plt
import func_data_process as proc
import metricx
# ######################################################################################
dir_data = 'data'
dir_mask = os.path.join(dir_data,'mask')

dir_results = 'results'
dir_results_simu = os.path.join(dir_results,'simu')
# --------------------------------------------------------------------------------------
id_test =  [100,101,102,103,104,105,106,107,108,109,
            110,111,112,113,114,115,    117,118,   
            120]
id_test_show = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,  19]
rescale     = [300.0,100.0]
sigma_simu  = 7.5

model_names = [ 'unettv_{}_{}'.format(0.0,'exp_l2norm_sse'),
                'unettv_{}_{}'.format(0.0,'sqexp_l2norm_sse'),
                'unettv_{}_{}'.format(0.0005,'sqexp_l2norm_sse'),
                # 'unettv_{}_{}'.format(0.003,'sqexp_l2norm_sse_tvp_l_0.1_new')]
                # 'unettv_{}_{}'.format(0.003,'sqexp_l2norm_sse_tvp_l_0.15_new')]
                'unettv_{}_{}'.format(0.003,'sqexp_l2norm_sse_tvp_l_0.1_bs32_a8p64s2_new')] # accept
                # 'unettv_{}_{}'.format(0.0025,'sqexp_l2norm_sse_tvp_l_0.15_bs32_a8p64s2')]

tes = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
methods = ['EXP','M$_2$NCM','M$_1$NCM','PCANR','UNet-EXP', 'UNet-M$_2$', 'UNet-TV', 'UNet-TVp']
# methods = ['EXP','M$_2$NCM','M$_1$NCM','PCANR','ResNet-EXP', 'ResNet-M$_2$', 'ResNet-TV', 'ResNet-TVp']

font_dir  = pathlib.Path(os.path.join('fonts','times.ttf'))
plt.rcParams['mathtext.fontset'] = "stix"

# ######################################################################################
# load results
# load mask
mask_body  = np.load(os.path.join(dir_mask,'mask_body.npy'))[id_test]
mask_liver = np.load(os.path.join(dir_mask,'mask_liver.npy'))[id_test]
mask_paren = np.load(os.path.join(dir_mask,'mask_parenchyma.npy'))[id_test]
print('> mask shape: ', mask_body.shape)
# --------------------------------------------------------------------------------------
# load results from conventional methods
map_exp_simu   = np.load(os.path.join(dir_results_simu,'map_simu_exp_{}.npy'.format(sigma_simu)))
map_m2ncm_simu = np.load(os.path.join(dir_results_simu,'map_simu_m2ncm_{}.npy'.format(sigma_simu)))
map_m1ncm_simu = np.load(os.path.join(dir_results_simu,'map_simu_m1ncm_{}.npy'.format(sigma_simu)))
map_pcanr_simu = np.load(os.path.join(dir_results_simu,'map_simu_pcanr_{}.npy'.format(sigma_simu)))

map_gt_simu = np.load(os.path.join(dir_results_simu,'map_simu_gt.npy'.format(sigma_simu)))
# --------------------------------------------------------------------------------------
# load results from u-nets
map_nets = []
for name in model_names:
    maps = np.load(os.path.join(dir_results_simu,name,'sigma_'+str(sigma_simu),'map.npy'))
    map_nets.append(maps)
# --------------------------------------------------------------------------------------
# load input images
imgs_simu = np.load(os.path.join(dir_results_simu,'imgs_{}.npy'.format(sigma_simu)))

# ######################################################################################
# preprocessing
imgs_simu   = imgs_simu[id_test_show]*rescale[0]

map_gt_simu = map_gt_simu[id_test_show]*rescale
map_gt_simu = proc.mask_out(map_gt_simu,mask_body)

map_exp_simu   = proc.mask_out(map_exp_simu[id_test_show],mask_body)
map_m2ncm_simu = proc.mask_out(map_m2ncm_simu[id_test_show],mask_body)
map_m1ncm_simu = proc.mask_out(map_m1ncm_simu[id_test_show],mask_body)
map_pcanr_simu = proc.mask_out(map_pcanr_simu[id_test_show],mask_body)

map_all = []
map_all.append(map_exp_simu)
map_all.append(map_m2ncm_simu)
map_all.append(map_m1ncm_simu)
map_all.append(map_pcanr_simu)

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

sig_test_alt  = 'two-sided'
sig_test_type = 'signed-rank'
# --------------------------------------------------------------------------------------
nrmses_p, ssims_p = [], []
for i in range(N_methods):
    pn,ps = [], []
    for j in range(N_methods):
        p_nrmse = metricx.Pvalue(nrmses[i],nrmses[j],alt=sig_test_alt,type=sig_test_type)
        p_ssim  = metricx.Pvalue(ssims[i], ssims[j], alt=sig_test_alt,type=sig_test_type)
        pn.append(p_nrmse)
        ps.append(p_ssim)
    nrmses_p.append(pn)
    ssims_p.append(ps)
# --------------------------------------------------------------------------------------
print('-'*98)
print('NRMSE and SSIM')
title = ['NRMSE (mean)','NRMSE (std)','SSIM (mean)','SSIM (std)']
data  = np.array([np.mean(np.array(nrmses),axis=-1), np.std(np.array(nrmses),axis=-1), np.mean(np.array(ssims),axis=-1), np.std(np.array(ssims),axis=-1)])
table = pandas.DataFrame(data,title,methods)        
print(table)

print('-'*98)
print('P value (NRMSE)')
table_p_nrmse = pandas.DataFrame(np.array(nrmses_p),methods,methods)
print(table_p_nrmse)

print('-'*98)
print('P value (SSIM)')
table_p_ssim  = pandas.DataFrame(np.array(ssims_p),methods,methods)
print(table_p_nrmse)
print('-'*98)

# ######################################################################################
# Mean R2* in ROI
mean_gt,_ = proc.mean_std_roi(map_gt_simu[...,1],mask_paren)
mean_all  = []
for mp in map_all:
    m,_ = proc.mean_std_roi(mp[...,1],mask_paren)
    mean_all.append(m)

# ######################################################################################
# parameter maps
i_sub = 0
tes   = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
# --------------------------------------------------------------------------------------
vmax_w = 500
vmax_p = mean_gt[i_sub]*1.5
vmax_p_diff = mean_gt[i_sub]*0.5
row, col = 5, 4
# left_x, left_y, right_x, right_y = 2, 9, 80, 9
left_x, left_y, right_x, right_y = 2, 9, 88, 9 # times
font_size = 8
# --------------------------------------------------------------------------------------
fig_width  = 7.16
fig_height = fig_width/8.0*4.67
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=row,ncols=col,figsize=(fig_width,fig_height),dpi=600,constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]

ax_wei = axes.ravel()[0:2]
ax_gt  = axes.ravel()[2:4]
ax_map = axes.ravel()[np.r_[4:8,12:16]]
ax_err = axes.ravel()[np.r_[8:12,16:20]]

# --------------------------------------------------------------------------------------
# Weighted images
# axes[0,0].text(x=left_x,y=left_y,s='T$_2^*$w',color='white',font=font_dir,fontsize=font_size)
# axes[0,1].text(x=left_x,y=left_y,s='T$_2^*$w',color='white',font=font_dir,fontsize=font_size)
for i_te,ax in enumerate(ax_wei):
    pcm_img = ax.imshow(imgs_simu[i_sub,:,:,i_te],cmap='gray',interpolation='none',vmin=0,vmax=vmax_w)
    # ax.text(x=right_x,y=right_y,s='TE$_{}$={}'.format(i_te+1,tes[i_te]),color='white',font=font_dir,fontsize=font_size)
# --------------------------------------------------------------------------------------
# Ground truth
# for j in [0,1]: ax_gt[j].text(x=left_x,y=left_y,s='GT',color='white',font=font_dir,fontsize=font_size)
ax_gt[0].imshow(map_gt_simu[i_sub,:,:,0],cmap='gray',interpolation='none',vmin=0,vmax=vmax_w)
# ax_gt[0].text(x=right_x+27,y=right_y,s='S$_0$',color='white',font=font_dir,fontsize=font_size)
ax_gt[1].imshow(map_gt_simu[i_sub,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p)
# ax_gt[1].text(x=right_x+27,y=right_y,s='R$_2^*$',color='white',font=font_dir,fontsize=font_size)
# --------------------------------------------------------------------------------------
# Reconstructed maps
for i_method in range(N_methods):
    pcm_map = ax_map[i_method].imshow(map_all[i_method][i_sub,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p)
    # ax_map[i_method].text(x=left_x,y=left_y,s=methods[i_method],color='white',font=font_dir,fontsize=font_size)
    # ax_map[i_method].text(x=right_x+10,y=right_y,s='{:.4f}'.format(ssims[i_method][i_sub]),color='white',font=font_dir,fontsize=font_size)
    
    pcm_dif = ax_err[i_method].imshow(np.abs(map_all[i_method][i_sub,:,:,1]-map_gt_simu[i_sub,:,:,1]),cmap='jet',interpolation='none',vmin=0,vmax=vmax_p_diff)
    # ax_err[i_method].text(x=left_x,y=left_y,s='Difference',color='white',font=font_dir,fontsize=font_size)
    # ax_err[i_method].text(x=right_x+10,y=right_y,s='{:.4f}'.format(nrmses[i_method][i_sub]),color='white',font=font_dir,fontsize=font_size)
# --------------------------------------------------------------------------------------
cbar1 = fig.colorbar(pcm_img,ax=axes[0,:],shrink=0.7,aspect=12,ticks=[0,vmax_w/2,vmax_w])
cbar2 = fig.colorbar(pcm_map,ax=axes[1,:],shrink=0.7,aspect=12,ticks=[0,500,vmax_p])
cbar3 = fig.colorbar(pcm_dif,ax=axes[2,:],shrink=0.7,aspect=12,ticks=[0,150,vmax_p_diff])
cbar4 = fig.colorbar(pcm_map,ax=axes[3,:],shrink=0.7,aspect=12,ticks=[0,500,vmax_p])
cbar5 = fig.colorbar(pcm_dif,ax=axes[4,:],shrink=0.7,aspect=12,ticks=[0,150,vmax_p_diff])
cbar1.ax.set_yticklabels(labels=[0,250,500],font=font_dir,fontsize=font_size)
cbar2.ax.set_yticklabels(labels=[0,500,900],font=font_dir,fontsize=font_size)
cbar3.ax.set_yticklabels(labels=[0,150,300],font=font_dir,fontsize=font_size)
cbar4.ax.set_yticklabels(labels=[0,500,900],font=font_dir,fontsize=font_size)
cbar5.ax.set_yticklabels(labels=[0,150,300],font=font_dir,fontsize=font_size)

plt.savefig(os.path.join('figures','map_test_simu_{}'.format(i_sub)))

# ######################################################################################
# Map profile
i_sub = 11
x,y   = 50,34
y_lim =np.max(map_gt_simu[i_sub,y,:,1])*1.1
line_width = 1.0
# --------------------------------------------------------------------------------------------------------
fig_width  = 7.16
fig_height = fig_width/3.0*2.0
# --------------------------------------------------------------------------------------------------------
fig,axes = plt.subplots(ncols=2,nrows=2,figsize=(fig_width,fig_height),dpi=600,constrained_layout=True)
for ax in [axes[0,0],axes[0,1]]: 
    ax.set_axis_off()
    ax.plot([0,127],[y,y],'k--')
    ax.plot([x,x],[0,63],'k--')

axes[0,0].imshow(map_gt_simu[i_sub,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p)
axes[0,0].set_title('GT')
axes[0,1].imshow(map_all[-1][i_sub,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p)
axes[0,1].set_title(methods[-1])
# --------------------------------------------------------------------------------------------------------
for ax in [axes[1,0],axes[1,1]]: ax.set_ylim((0,y_lim))
axes[1,0].set_xlim((0,128))
axes[1,1].set_xlim((0,64))

axes[1,0].set_title('Horizontal Line')
axes[1,0].plot(map_gt_simu[i_sub,y,:,1],'k--',label='Reference',linewidth = line_width)
axes[1,0].plot(map_all[3][i_sub,y,:,1],'--.', label=methods[3],linewidth = line_width)
axes[1,0].plot(map_all[6][i_sub,y,:,1],'--.', label=methods[6],linewidth = line_width)
axes[1,0].plot(map_all[7][i_sub,y,:,1],'r--.',label=methods[7],linewidth = line_width)
axes[1,0].tick_params(direction='in')
axes[1,0].legend(fontsize=8)

axes[1,1].set_title('Vertical Line')
axes[1,1].plot(map_gt_simu[i_sub,:,x,1],'k--',label='Reference',linewidth = line_width)
axes[1,1].plot(map_all[3][i_sub,:,x,1],'--.', label=methods[3],linewidth = line_width)
axes[1,1].plot(map_all[6][i_sub,:,x,1],'--.', label=methods[6],linewidth = line_width)
axes[1,1].plot(map_all[7][i_sub,:,x,1],'r--.',label=methods[7],linewidth = line_width)
axes[1,1].tick_params(direction='in')
plt.savefig(os.path.join('figures','map_profile_{}'.format(i)))

# ######################################################################################
# Bland-Altman Analysis
print('-'*98)
print('Bland-Altman plot Analysing...')
xLimit,yLimit = 1100,100
row, col   = 2, 4
line_width = 1.0
font_size  = 7.5
fig_width  = 7.16
fig_height = fig_width/2.5
# --------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=row,ncols=col,figsize=(fig_width,fig_height),dpi=600,constrained_layout=True)
axs = axes.ravel()

ref_name = 'Reference'
y = mean_gt

for k in range(N_methods):
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
        axs[k].text(x=850,y=md+1.96*sd+20,s='{:>6.2f}'.format(md+1.96*sd),font=font_dir,fontsize=font_size,color='red')

    if np.abs(md-1.96*sd) < yLimit:
        axs[k].text(x=40, y=md-1.96*sd-20,s='-1.96SD',font=font_dir,fontsize=font_size,color='black')
        axs[k].text(x=850,y=md-1.96*sd-20,s='{:>6.2f}'.format(md-1.96*sd),font=font_dir,fontsize=font_size,color='red')

    axs[k].text(x=850,y=md+3,s='{:6.2f}'.format(md),font=font_dir,fontsize=font_size,color='red')

    axs[k].set_ylim([-yLimit,yLimit])
    axs[k].set_ylabel('{} - {} (s$^{{-1}}$)'.format(methods[k],ref_name),font=font_dir,fontsize=font_size)
    axs[k].set_xlim([0,xLimit])
    axs[k].set_xlabel('({} + {})/2 (s$^{{-1}}$)'.format(methods[k],ref_name),font=font_dir,fontsize=font_size)
    axs[k].set_xticks(ticks=[0,250,500,750,1000])
    axs[k].set_xticklabels(labels=[0,250,500,750,1000],font=font_dir)
    axs[k].set_yticks(ticks=[-100,-50,0,50,100])
    axs[k].set_yticklabels(labels=[-100,-50,0,50,100],font=font_dir)
    axs[k].tick_params(axis='both',direction='in',length=2.0,labelsize=7.5)

plt.savefig(os.path.join('figures','baplot_simu.png'))

# ######################################################################################
# NRMSE and SSIM Bars
fig_width  = 3.5
fig_height = fig_width/2.0
font_size  = 6
# --------------------------------------------------------------------------------------
plt.figure(figsize=(fig_width,fig_height),dpi=600,constrained_layout=True)
ind = np.argsort(mean_gt)
i   = np.linspace(0,mean_gt.shape[0]-1,mean_gt.shape[0])
for i_method in range(N_methods):
    plt.bar(i+0.1*i_method,nrmses[i_method][ind], width=0.1,label=methods[i_method])
plt.legend(fontsize=font_size,ncol=2)
plt.tick_params(labelsize=font_size)
plt.savefig(os.path.join('figures','NRMSEbar.png'))
# --------------------------------------------------------------------------------------
plt.figure(figsize=(fig_width,fig_height),dpi=600,constrained_layout=True)
ind = np.argsort(mean_gt)
i   = np.linspace(0,mean_gt.shape[0]-1,mean_gt.shape[0])
for i_method in range(N_methods):
    plt.bar(i+0.1*i_method,ssims[i_method][ind], width=0.1,label=methods[i_method])
plt.ylim(0.5,1.0)
plt.legend(fontsize=font_size,ncol=2)
plt.tick_params(labelsize=font_size)
plt.savefig(os.path.join('figures','SSIMbar.png'))
# ######################################################################################
print(np.round(mean_gt[ind],1))
print(np.round(mean_gt,1))
print('='*98)