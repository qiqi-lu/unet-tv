import os
import numpy as np
import tensorflow as tf
import func_data_process as proc
import config

config.config_gpu(5)

######################################################################################
dir_data = 'data'
dir_data_iv = os.path.join(dir_data,'invivo')

dir_result = 'results'
dir_result_invivo = os.path.join(dir_result,'invivo')

rescale     = [300.0,100.0]

######################################################################################
# load data
print('='*98)
print('load datasets ...')
datasets_filename = tf.io.gfile.glob(os.path.join(dir_data_iv,'test_invivo.tfrecords'))
datasets = tf.data.TFRecordDataset(datasets_filename).map(proc.parse_all)\
                 .map(lambda x1,x2,x3,x4: proc.extract(x1,x2,x3,x4,rescale=rescale,model_type='physic-inform'))
datasets_size = proc.get_len(datasets)
datasets = datasets.batch(datasets_size).take(1)

# input data
for sample in datasets: imgs_iv = sample[0][0].numpy()
print('Dataset shape (in vivo): ',imgs_iv.shape)

######################################################################################
# load model
print('='*98)
print('load models ...')
# model_name = 'unettv_{}_{}'.format(0.0,'exp_l2norm_sse')
# model_name = 'unettv_{}_{}'.format(0.0,'sqexp_l2norm_sse')
# model_name = 'unettv_{}_{}'.format(0.001,'sqexp_l2norm_sse')
# model_name = 'unettv_{}_{}'.format(0.003,'sqexp_l2norm_sse_tvp_l_0.1_new')
model_name = 'unettv_{}_{}'.format(0.003,'sqexp_l2norm_sse_tvp_l_0.1_bs32_a8p64s2_new')
epoch = '200'
# -----------------------------------------------------------------------------------
model_dir = os.path.join('model','unet',model_name,'model_{}.h5'.format(epoch))
model = tf.keras.models.load_model(model_dir,compile=False)
print(model_dir)
print('predict ...')
xp,mapp = model.predict(datasets)

######################################################################################
# save
name = os.path.join(dir_result_invivo,model_name)
if not os.path.exists(name): os.mkdir(name)

np.save(file=os.path.join(dir_result_invivo,'imgs'),arr=imgs_iv)
np.save(file=os.path.join(name,'map'),arr=mapp)
np.save(file=os.path.join(name,'xp'),arr=xp)

######################################################################################
print('='*98)