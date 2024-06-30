import tensorflow as tf
import os
import models as mod
import func_model as func
import config
import func_data_process as proc

config.config_gpu(0)
#############################################################################################
rescale    = [300.0,100.0]
model_type = 'sqexp'
suffix     = 'l2norm_sse_tvp_l_0.15_bs32_a8p64s2'
weight     = 0.0025
batch_size = 32
epochs     = 300
save_every = 20
validation_batch_size = 32
lr = 0.0001

dir_data_iv = os.path.join('data','invivo')
#############################################################################################
## Dataset
print('='*98)
print('Load dataset ...')
dataset_filenames = tf.io.gfile.glob(os.path.join(dir_data_iv,'train_invivo_a8p64s2.tfrecords'))
dataset = tf.data.TFRecordDataset(dataset_filenames).map(proc.parse_all)\
                .map(lambda x1,x2,x3,x4: proc.extract(x1,x2,x3,x4,rescale=rescale,model_type='physic-inform'))

dataset_size = proc.get_len(dataset)
split = 0.8
dataset_train_size = int(dataset_size*split)
dataset_valid_size = int(dataset_size*(1.0-split))

dataset_valid = dataset.shard(num_shards=5,index=3)

dataset_train = dataset.shard(num_shards=5,index=0)
dataset_train = dataset_train.concatenate(dataset.shard(num_shards=5,index=1))
dataset_train = dataset_train.concatenate(dataset.shard(num_shards=5,index=4))
dataset_train = dataset_train.concatenate(dataset.shard(num_shards=5,index=2))

dataset_train.shuffle(1000,reshuffle_each_iteration=True,seed=8200)
dataset_valid.shuffle(1000,reshuffle_each_iteration=True,seed=8200)

print('Dataset:')
print(dataset_filenames)
print('  Training data size : '+str(dataset_train_size))
print('Validation data size : '+str(dataset_valid_size))
print('-'*98)

#############################################################################################
## MODEL
def sqexp(x):
    tes    = x[2][...,tf.newaxis,tf.newaxis,:]
    m0,r2  = tf.split(x[0],2,axis=-1)
    reconv = tf.math.multiply(tf.math.square(m0),tf.math.exp(-2.0*tes/1000.0*r2)) + 2.0*tf.math.square(x[1])
    reconv = tf.math.sqrt(reconv)
    return reconv

def exp(x):
    tes    = x[2][...,tf.newaxis,tf.newaxis,:]
    m0,r2  = tf.split(x[0],2,axis=-1)
    reconv = tf.math.multiply(m0,tf.math.exp(-1.0*tes/1000.0*r2))
    return reconv

def constraint(x):
    p0,p1 = tf.split(x,2,axis=-1)
    p0    = tf.clip_by_value(p0,0.0,4.0)
    p1    = tf.clip_by_value(p1,0.0,15.0)
    out   = tf.concat([p0,p1],axis=-1)
    return out

inpts_img = tf.keras.Input(shape=(None,None,12))
inpts_sg  = tf.keras.Input(shape=(None,None,1))
inpts_tes = tf.keras.Input(shape=(12,))
# model_mapping = mod.UNet(image_channels=12,output_channel=2)
model_mapping = mod.ResNet_moled(image_channels=12,output_channel=2,num_block=4)
outpts_map    = model_mapping(inpts_img)

outpts_map    = tf.keras.layers.Lambda(constraint,name='map')(outpts_map)

if model_type == 'exp':   outpts_img = tf.keras.layers.Lambda(exp,name='img')([outpts_map,inpts_sg,inpts_tes])
if model_type == 'sqexp': outpts_img = tf.keras.layers.Lambda(sqexp,name='img')([outpts_map,inpts_sg,inpts_tes])
model = tf.keras.Model(inputs=(inpts_img,inpts_sg,inpts_tes),outputs=(outpts_img,outpts_map))
model.summary()

#############################################################################################
# loss_img = tf.keras.losses.MeanSquaredError()
# loss_img = tf.keras.losses.MeanAbsoluteError()
# loss_img = func.l2norm
loss_img = func.l2norm_sse
# loss_img = func.l1norm
# loss_img = func.l1norm_sae

# loss_map = func.dummy
# loss_map = func.tv
loss_map = func.tvp

#############################################################################################
## TRAINING
steps_per_epoch  = tf.math.floor(dataset_train_size/batch_size)
validation_steps = tf.math.floor(dataset_valid_size/validation_batch_size)

model_dir     = os.path.join('model','unet','unettv_{}_{}_{}'.format(weight,model_type,suffix))
initial_epoch = func.findLastCheckpoint(save_dir=model_dir)
if initial_epoch > 0:  
    print('Resuming by loading epoch %03d'%initial_epoch)
    model = tf.keras.models.load_model(os.path.join(model_dir,'model_%03d.h5'%initial_epoch), compile=False)

opti = tf.keras.optimizers.Adam(learning_rate=lr)

model.compile(optimizer=opti, loss=[loss_img,loss_map], loss_weights=[1.0,weight])

checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir,'model_{epoch:03d}.h5'), verbose=1, save_weights_only=False, period=save_every)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(func.lr_schedule)
tensorboard  = tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=1)

model.fit(  x = dataset_train.repeat(epochs).batch(batch_size,drop_remainder=True),
            epochs           = epochs,
            steps_per_epoch  = steps_per_epoch,
            validation_data  = dataset_valid.batch(validation_batch_size,drop_remainder=True),
            validation_steps = validation_steps,
            initial_epoch    = initial_epoch,
            callbacks        = [checkpointer,tensorboard,lr_scheduler],
            # callbacks        = [checkpointer,tensorboard],
            )