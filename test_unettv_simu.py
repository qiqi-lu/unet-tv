import os
import numpy as np
import tensorflow as tf
import func_data_process as proc
import config

config.config_gpu(0)
##########################################################################################################
dir_data      = 'data'
dir_data_simu = os.path.join('data','simu')
dir_results   = 'results'
dir_results_simu = os.path.join(dir_results,'simu')

sigma_simu = 7.5
rescale    = [300.0,100.0]

# -----------------------------------------------------------------------------------
# model_name = 'unettv_{}_{}'.format(0.0,'exp_l2norm_sse')
# model_name = 'unettv_{}_{}'.format(0.0,'sqexp_l2norm_sse')
# model_name = 'unettv_{}_{}'.format(0.001,'sqexp_l2norm_sse')
# model_name = 'unettv_{}_{}'.format(0.003,'sqexp_l2norm_sse_tvp_l_0.1_new')
model_name = 'unettv_{}_{}'.format(0.0025,'sqexp_l2norm_sse_tvp_l_0.15_bs32_a8p64s2')

epoch = '300'
##########################################################################################################
print('='*98)
print('Load models ...')
model_dir = os.path.join('model','unet',model_name,'model_{}.h5'.format(epoch))
print(model_dir)
model = tf.keras.models.load_model(model_dir,compile=False)
# -----------------------------------------------------------------------------------
dataset_filenames_simu= tf.io.gfile.glob(os.path.join(dir_data_simu,'test_simu_{}.tfrecords'.format(sigma_simu)))
dataset_simu          = tf.data.TFRecordDataset(dataset_filenames_simu).map(proc.parse_all)\
                            .map(lambda x1,x2,x3,x4: proc.extract(x1,x2,x3,x4,rescale=rescale,model_type='physic-inform'))
dataset_simu_size = proc.get_len(dataset_simu)
dataset_simu      = dataset_simu.batch(dataset_simu_size).take(1)

for sample in dataset_simu:
    imgs_simu   = sample[0][0].numpy()
    map_gt_simu = sample[1][1].numpy()
print('Dataset (simulation):')
print('> Image: ',imgs_simu.shape)
print('> Map:   ',map_gt_simu.shape)

xp,mapp = model.predict(dataset_simu)

######################################################################################
# save
name = os.path.join(dir_results_simu,model_name,'sigma_'+str(sigma_simu))
if not os.path.exists(name): os.makedirs(name)

print('Save to ', name)
np.save(file=os.path.join(dir_results_simu,'imgs_{}'.format(sigma_simu)),arr=imgs_simu)
np.save(file=os.path.join(dir_results_simu,'map_simu_gt'),arr=map_gt_simu)
np.save(file=os.path.join(name,'map'),arr=mapp)
np.save(file=os.path.join(name,'xp'),arr=xp)

######################################################################################
print('='*98)