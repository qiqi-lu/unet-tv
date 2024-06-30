import tensorflow as tf
import glob, re, os

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.h5'))  # get name list of all .hdf5 files
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).h5.*",file_)
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def lr_schedule(epoch):
    initial_lr = 0.0005
    if epoch<=10:
        lr = initial_lr
    elif epoch<=20:
        lr = initial_lr/2
    elif epoch<=40:
        lr = initial_lr/4 
    elif epoch<=80:
        lr = initial_lr/8
    elif epoch<=160:
        lr = initial_lr/16
    else:
        lr = initial_lr/32 
    return lr

def NRMSE(y_true,y_pred):
    e       = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_pred,y_true)),axis=[1,2]))
    gt      = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_true),axis=[1,2]))
    nrmse   = tf.math.reduce_mean(tf.math.divide_no_nan(e,gt))
    return nrmse

def tvp(y_true,y_pred):
    a = 0.15
    p = 2.0

    eps  = 1e-6
    pad1 = tf.constant([[0,0],[0,1],[0,0],[0,0]])
    pad2 = tf.constant([[0,0],[0,0],[0,1],[0,0]])
    pixel_dif1 = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :] # finite forward difference donw->up
    pixel_dif2 = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :] # right->left
    pixel_dif1 = tf.pad(pixel_dif1,pad1,"CONSTANT")
    pixel_dif2 = tf.pad(pixel_dif2,pad2,"CONSTANT")

    tv = tf.math.sqrt(tf.math.square(pixel_dif1)+tf.math.square(pixel_dif2)+tf.math.square(eps))

    # tv = tf.where(tv<=a, 1.0/(p*a)*tf.math.square(tv), a*tf.math.log(tv)+1.0/p*a-a*tf.math.log(a))
    tv = tf.where(tv<=a, tv, a*tf.math.log(tv)+2.0/p*a-a*tf.math.log(a))

    tv = tf.reduce_mean(tf.math.reduce_sum(tv,axis=[1,2,3]))
    # tv = tf.reduce_mean(tv)
    return tv

def tv(y_true,y_pred):
    eps  = 1e-6
    pad1 = tf.constant([[0,0],[0,1],[0,0],[0,0]])
    pad2 = tf.constant([[0,0],[0,0],[0,1],[0,0]])
    pixel_dif1 = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :] # finite forward difference donw->up
    pixel_dif2 = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :] # right->left
    pixel_dif1 = tf.pad(pixel_dif1,pad1,"CONSTANT")
    pixel_dif2 = tf.pad(pixel_dif2,pad2,"CONSTANT")

    tv = tf.math.sqrt(tf.math.square(pixel_dif1)+tf.math.square(pixel_dif2)+tf.math.square(eps))
    tv = tf.reduce_mean(tf.math.reduce_sum(tv,axis=[1,2,3]))
    # tv = tf.reduce_mean(tf.math.reduce_sum(tv,axis=-1))
    # tv = tf.reduce_mean(tv)
    return tv

def l2norm(y_true,y_pred):
    # mse
    loss = tf.math.reduce_sum(tf.math.square(y_pred-y_true),axis=-1)
    loss = tf.math.reduce_mean(loss)
    return loss

def l2norm_sse(y_true,y_pred):
    loss = tf.math.reduce_sum(tf.math.square(y_pred-y_true),axis=[1,2,3])
    loss = tf.math.reduce_mean(loss)
    return loss

def l1norm(y_true,y_pred):
    loss = tf.math.reduce_sum(tf.math.abs(y_pred-y_true),axis=-1)
    loss = tf.math.reduce_mean(loss)
    return loss

def l1norm_sae(y_true,y_pred):
    loss = tf.math.reduce_sum(tf.math.abs(y_pred-y_true),axis=[1,2,3])
    loss = tf.math.reduce_mean(loss)
    return loss

def dummy(y_true,y_pred):
    return [0.0,1.0]