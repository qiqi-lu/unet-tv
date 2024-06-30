# functions for data process
import glob
import numpy as np
import pydicom
import os
import SimpleITK as sitk
import sys
import tqdm
import matplotlib.pyplot as plt
import re
import cv2
import methods
import tensorflow as tf
import multiprocessing
import operator

def read_dicom(file_dir):
    """ 
    Read all .dcm file in the folder into a array and their TE value
    """
    file_dir    = os.path.join(file_dir,'*.dcm')
    file_names  = sorted(glob.glob(file_dir),key=os.path.getmtime,reverse=True)
    num_files   = np.array(file_names).shape[0]
    imgs,tes    = [],[]
    for file,_ in zip(file_names, range(0,num_files)):
        imgs.append(pydicom.dcmread(file).pixel_array)
        tes.append(pydicom.dcmread(file).EchoTime)
    imgs = np.array(imgs).transpose((1,2,0))
    return imgs,np.array(tes)

def read_dicom_itk(file_dir,shape=(64,128,12)):
    """
    Use itk to read raw dicom data and resampled to the same size.
    - file_dir, xxx/study*.
    """
    name_folders= sorted(glob.glob(file_dir),key=os.path.getmtime,reverse=False)
    num_study   = np.array(name_folders).shape[0]
    data_study  = np.zeros([num_study,shape[0],shape[1],shape[2]])

    process_bar = tqdm.tqdm(desc='Read DICOM: ',total=num_study)
    for id_study in range(num_study):
        process_bar.update()
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(name_folders[id_study])
        if not series_ids:
            print("ERROR: given directory dose not a DICOM series.")
            sys.exit(1)
         
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(name_folders[id_study],series_ids[0])
        series_reader     = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        images = series_reader.Execute()
        # resahpe image into same size
        images_resample = resample_sitk_image_series(out_size=[data_study.shape[-2],data_study.shape[-3]],imgs_sitk=images)
        images_array    = sitk.GetArrayFromImage(images_resample)
        for id_image in range(data_study.shape[-1]):
            data_study[id_study,:,:,id_image] = images_array[id_image,:,:]
    process_bar.close()
    return data_study

def data_info(study_path,show=False):
    # read data information
    study_names = sorted(glob.glob(study_path),key=os.path.getmtime,reverse=True)
    num_study   = np.array(study_names).shape[0]
    TE,Sex,Name,Age,AcquisitionDate,SequenceName,B0,Manu,Institution,ID=[],[],[],[],[],[],[],[],[],[]
    TR,FA,ST,PS,Matrix,Ave,FS = [],[],[],[],[],[],[]
    num_female,num_male = 0,0

    # for every study
    print('Read data information ...')
    if show: print('Study name | B0 | Repetition time | Flip angle | TEs | Slice thickness | Resolution | Matrix | Ave | FatSat | Manu | Sequence')
    for study_name,id_study in zip(study_names, range(0,num_study)):
        # get all dcm data in this study
        image_names = glob.glob(os.path.join(study_name, '*.dcm'))
        image_names = sorted(image_names,key=os.path.getmtime,reverse=True)
        num_files   = np.array(image_names).shape[0]

        # get echo times
        tes = []
        for image_name,i in zip(image_names, range(0,num_files)):
            x = pydicom.dcmread(image_name)
            tes.append(x.EchoTime)
        TE.append(tes)

        # get other information
        X   = pydicom.dcmread(image_names[0])
        age = re.findall('0(\d+)Y',X.PatientAge)
        age = list(map(int,age))[0]
        if X.PatientSex=='F':
            sex=2
            num_female = num_female+1
        else:
            sex=1
            num_male = num_male+1
        Sex.append(sex),Name.append(X.PatientName),Age.append(age)
        AcquisitionDate.append(X.AcquisitionDate),SequenceName.append(X.SequenceName)
        Manu.append(X.Manufacturer),B0.append(X.MagneticFieldStrength)

        TR.append(X.RepetitionTime),FA.append(X.FlipAngle),ST.append(X.SliceThickness)
        PS.append(X.PixelSpacing),Matrix.append(X.AcquisitionMatrix),Ave.append(X.NumberOfAverages)
        FS.append(X.ScanOptions)
        try:
            Institution.append(X.InstitutionName)
        except:
            Institution.append('None') # may some study have not saved institution name
        ID.append(id_study)
    
    if show:
        for i in range(num_study):
            print('{:>3d} | {:.2f} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}'.format(
                ID[i],B0[i],TR[i],FA[i],TE[i],ST[i],np.round(PS[i],decimals=3),Matrix[i],Ave[i],FS[i],Manu[i],SequenceName[i]))

    # Age distribution
    age_mean,age_std,age_max,age_min = np.mean(Age),np.std(Age),np.max(Age),np.min(Age)
    print('> Age: {:>.1f}(+-{:>.1f}) [{:>.1f},{:>.1f}], Sex: {:>.1f}(female) | {:>.1f}(male)'.format(age_mean,age_std,age_min,age_max,num_female,num_male))
     
    info = {'study_names':study_names,'TE':TE,'Sex':Sex,'Name':Name,'Age':Age,'AcquisitionData':AcquisitionDate,
            'SequenceName':SequenceName,'B0':B0,'Manu':Manu,'Institution':Institution,'ID':ID}
    return info

def resample_sitk_image_series(out_size,imgs_sitk):
    """Resample image to specific size, the pixel size has been changed!
    """
    input_size    = imgs_sitk.GetSize()
    input_spacing = imgs_sitk.GetSpacing()
    output_size   = (out_size[0],out_size[1],input_size[2])
    
    output_spacing    = np.array([0.,0.,0.]).astype('float64')
    output_spacing[0] = input_size[0]*input_spacing[0]/output_size[0]
    output_spacing[1] = input_size[1]*input_spacing[1]/output_size[1]
    output_spacing[2] = input_size[2]*input_spacing[2]/output_size[2]
    
    transform = sitk.Transform()
    transform.SetIdentity()

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(imgs_sitk.GetOrigin())
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputDirection(imgs_sitk.GetDirection())
    resampler.SetSize(output_size)
    imgs_sitk_resample = resampler.Execute(imgs_sitk)
    return imgs_sitk_resample

def sigma_bkg(data,mask=None,num_coil=1,mean=True):
    """
    Calculate the mean of the bkg in [each study, each TE]
    ### ARGUMENTS
    - data, [Ny,Nx, Nq] or [N,Ny,Nx,Nq]
    - mask, the mask out region.
    - mean, mean along the te axis, if `mean=True`, return the sigma of each study [N].

    ### RETURN
    - sigma_g: sigma map of noise in each TE in each study. [N, Nq]
    """
    print('Background noise sigma estimation ...')
    if len(data.shape) == 3: data=data[np.newaxis]    
    n,h,w,c = data.shape
    if mask is None:
        f=5 # roi size
        roi1 = np.zeros((n,c))
        roi2 = np.zeros((n,c))
        for i in range(n): # each study
            for j in range(c): # each TE time
                roi1[i,j] = np.mean(data[i,1:1+f,1:1+f,j]**2)
                roi2[i,j] = np.mean(data[i,1:1+f,w-f-1:w-1+f,j]**2)
        mean_bkg = np.mean([roi1,roi2],axis=0)
    if mask is not None:
        mask = (mask+1)%2 # convert 1 to 0.
        kernel = np.ones((5,5))
        for i in range(n):
            mask[i] = cv2.erode(mask[i],kernel=kernel)
        mask = np.repeat(mask[...,np.newaxis],c,axis=-1)
        mean_bkg = np.sum(np.sum((data*mask)**2,axis=1),axis=1)/np.sum(np.sum(mask,axis=1),axis=1)
    sigma_g  = np.sqrt(mean_bkg/(2*num_coil))
    if mean==True: sigma_g = np.mean(sigma_g,axis=1)
    return sigma_g

def image2map(imgs,tes,method='PW',model='EXP',mask=None,num_coils=1,fix_sigma=True):
    """
    Parameter mapping.
    #### ARGUMENTS
    - imgs, weigthed images. [N,Ny,Nx,Nq]
    - tes, echo times. [Nq]

    #### RETURN
    - maps, parameter maps. [N,Ny,Nx,Np]
    """
    sigma = sigma_bkg(imgs,num_coil=num_coils, mask=mask,mean=True)
    maps = []
    N    = imgs.shape[0]
    if method == 'PW': # Pixel-wise fitting
        pbar = tqdm.tqdm(desc=method+'-'+model,total=N,leave=True)
        for i in range(N):
            pbar.update(1)
            if fix_sigma == True:  map = methods.PixelWiseMapping(imgs[i],tes=tes,model=model,sigma=sigma[i],NCoils=num_coils,filtering=None,pbar_leave=False)
            if fix_sigma == False: map = methods.PixelWiseMapping(imgs[i],tes=tes,model=model,sigma=None,NCoils=num_coils,filtering=None,pbar_leave=False)
            maps.append(map)
        pbar.close()

    if method == 'PCANR':
        global func
        def func(i):
            if fix_sigma == True:  map = methods.PCANR(imgs[i],tes=tes,sigma=sigma[i],beta=1.2,f=5,m=0,Ncoils=num_coils,model=model,pbar_disable=True)
            if fix_sigma == False: map = methods.PCANR(imgs[i],tes=tes,sigma=None,beta=0.7,f=5,m=0,Ncoils=num_coils,model=model,pbar_disable=True)
            return (i,map)
        accelerate_factor = 4
        pool = multiprocessing.Pool(accelerate_factor)
        pbar = tqdm.tqdm(desc=method+'-'+model,total=N,leave=True)
        for kp in pool.imap_unordered(func,range(N)):
            maps.append(kp)
            pbar.update(1)
        pool.close()
        pbar.close()
        maps = sorted(maps,key=operator.itemgetter(0))
        maps = [p[1] for p in maps]


    if method == 'ATF': # average then fitting
        pbar = tqdm.tqdm(desc=method+'-'+model,total=N,leave=True)
        for i in range(N):
            pbar.update(1)
            if fix_sigma == True:  map = methods.AverageThenFitting(imgs[i],tes=tes,mask=mask,model=model,sigma=sigma[i],NCoils=num_coils)
            if fix_sigma == False: map = methods.AverageThenFitting(imgs[i],tes=tes,mask=mask,model=model,sigma=None,NCoils=num_coils)
            maps.append(map)
            pbar.close()
    maps = np.stack(maps)
    print('maps: '+ str(maps.shape))
    return maps

def map2image(maps,tes,map_type='R2'):
    """
    Parameter maps to weigthed images (simulated).
    #### ARGUMENTS
    - maps, parameter maps, [N,Ny,Nx,Np]
    - tes, echo times. [Nq]
    
    #### RETURN
    - imgs, weigthed images. [N,Ny,Nx,Nq]
    """
    N,Ny,Nx,Np = maps.shape
    Nq   = tes.shape[-1]
    imgs = np.zeros(shape=(N,Ny,Nx,Nq))
    pbar = tqdm.tqdm(desc='Relax',total=N,leave=True)
    for n in range(N):
        pbar.update(1)
        for i in range(Ny):
            for j in range(Nx):
                if map_type=='R2':
                    imgs[n,i,j] = methods.S_R2(M0=maps[n,i,j,0],R2=maps[n,i,j,1],TEs=tes)
                if map_type=='T2':
                    imgs[n,i,j] = methods.S_T2(M0=maps[n,i,j,0],T2=maps[n,i,j,1],TEs=tes)
    pbar.close()
    print('imgs (nf): '+ str(imgs.shape))
    return imgs

def mask_out(imgs,mask):
    return imgs*mask[...,np.newaxis]

def patch(data,mask,patch_size=32,step_size=8,aug=1,id=False):
    """ 
    Generate patches.
    #### ARGUMENTS
    - data, [Nz,Ny,Nx,Nc]
    - mask, [Nz,Ny,Nx]

    #### RETURN
    - patches, [N,Ny,Nx,Nc].
    """
    Nz,Ny,Nx,_ = data.shape
    num_step_x = (Nx-patch_size)//step_size
    num_step_y = (Ny-patch_size)//step_size
    patches,ids= [],[]
    pbar=tqdm.tqdm(desc='Patching',total=(num_step_x+1)*(num_step_y+1)*Nz)
    for k in range(Nz):
        for i in range(num_step_y+1):
            for j in range(num_step_x+1):
                pbar.update(1)
                if mask[k,i*step_size+int(patch_size/2),j*step_size+int(patch_size/2)]>0:
                    patch = data[k,i*step_size:i*step_size+patch_size,j*step_size:j*step_size+patch_size,:]
                    for mod in range(aug):
                        patch = augmentation(patch,mode=mod)
                        patches.append(patch)
                        ids.append(k)
    pbar.close()
    patches = np.stack(patches)
    ids     = np.stack(ids)
    print(patches.shape)
    if id == True:  return patches,ids
    if id == False: return patches

def addNoise(imgs,sigma,noise_type='Rician',NCoils=1,random_seed=0):
    """
    Add nosie to the inputs with fixed standard deviation.
    ### ARGUMENTS
    - imgs, the image to be added noise.
    - sigma, noise sigma.
    - noise_type, type of noise.

    ### RETURN
    - imgs_n: images with noise.
    """
    np.random.seed(seed=random_seed)
    type = ['Rician', 'Gaussian','ChiSquare']
    assert sigma>0, 'Noise sigma must higher than 0.'
    assert noise_type in type, 'Unsupported noise type.'
    assert NCoils > 0, 'Coils number should larger than 0.'

    print('Add '+noise_type+' noise... (sigma = '+str(sigma)+')')
    if noise_type == 'Gaussian':
        imgs_n = imgs + np.random.normal(loc=0,scale=sigma,size=imgs.shape)
    if noise_type == 'Rician':
        r = imgs + np.random.normal(loc=0,scale=sigma,size=imgs.shape)
        i = np.random.normal(loc=0,scale=sigma,size=imgs.shape)
        imgs_n = np.sqrt(r**2+i**2)
    if noise_type == 'ChiSquare':
        imgs = imgs/np.sqrt(NCoils)
        imgs_n = np.zeros(imgs.shape)
        for i in range(NCoils):
            r = imgs+np.random.normal(loc=0,scale=sigma,size=imgs.shape)
            i = np.random.normal(loc=0,scale=sigma,size=imgs.shape)
            imgs_n = imgs_n+r**2+i**2
        imgs_n = np.sqrt(imgs_n)
    return imgs_n

def addNoiseMix(imgs,sigma_low,sigma_high,noise_type='Gaussian'):
    """
    Add nosie to the inputs with mixed standard deviation.
    ### ARGUMENTS
    - imgs, [N,Nx,Ny,num_channel].
    - sigma_low, sigma low bounding.
    - sigma_high, sigma high bounding.
    - noise_type, type of noise.

    ### RETURN
    - imgs_n: images with noise.
    """
    type = ['Rician', 'Gaussian']
    assert noise_type in type, 'Unsupported noise type.'
    N = imgs.shape[0]
    imgs_n = np.zeros_like(imgs)
    pbar = tqdm.tqdm(total=N,desc='Add Noise (mix)')
    for i in range(N):
        pbar.update(1)
        sigma = np.random.uniform(low=sigma_low,high=sigma_high)
        img   = imgs[i]
        if noise_type == 'Gaussian':
            noise = np.random.normal(loc=0.0,scale=sigma,size=img.shape)
            noise = np.maximum(np.minimum(noise,2.0*sigma),-2.0*sigma)
            imgs_n[i] = img + noise
        if noise_type == 'Rician':
            noise_r = np.random.normal(loc=0.0,scale=sigma,size=img.shape)
            noise_r = np.maximum(np.minimum(noise_r,2.0*sigma),-2.0*sigma)
            r = img + noise_r
            i = np.random.normal(loc=0.0,scale=sigma,size=img.shape)
            i = np.maximum(np.minimum(i,2.0*sigma),-2.0*sigma)
            imgs_n[i] = np.sqrt(r**2+i**2)
    pbar.close()
    return imgs_n

def roi_average(imgs,mask):
    mask = mask[...,np.newaxis]
    imgs_mask = imgs*mask
    s_ave = np.sum(imgs_mask,axis=(1,2))/np.sum(mask,axis=(1,2))
    return s_ave

def rearrange(data,idx):
    data_re = []
    for i in range(idx.shape[0]):
        data_re.append(data[idx[i]])
    data_re = np.stack(data_re)
    return data_re

def augmentation(img, mode=0):
    # aug data size
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

# The following function can be used to convert a value to a type comptible with tf.train.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write2TFRecord(patches_imgs,patches_maps,tes,sigma,filename):
    """
    Write numpy array to a TFRecord file.
    ### ARGUMENTS
    - patches_imgs, [N,Nx,Ny,Nc].
    - patches_maps, [N,Nx,Ny,Np].
    - tes,          [N,Nc]
    - sigma,        [N,Nx,Ny,1]
    - filename, `filename`.tfrecords.
    """
    filename = filename+'.tfrecords'
    writer = tf.io.TFRecordWriter(filename) # writer taht will store data to disk
    N,Nx,Ny,Nc = patches_imgs.shape
    _,_,_,Np   = patches_maps.shape

    for i in range(N):
        feature = {
            'patches_imgs': _bytes_feature(tf.io.serialize_tensor(patches_imgs[i])),
            'patches_maps': _bytes_feature(tf.io.serialize_tensor(patches_maps[i])),
            'tes':          _bytes_feature(tf.io.serialize_tensor(tes[i])),
            'sigma':        _bytes_feature(tf.io.serialize_tensor(sigma[i])),
            'Nx': _int64_feature(Nx),
            'Ny': _int64_feature(Ny),
            'Nc': _int64_feature(Nc),
            'Np': _int64_feature(Np),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        example = example.SerializeToString()
        writer.write(example)
    writer.close()
    print('Wrote '+str(N)+' elemets to TFRecord')

def parse_all(example):
    feature_discription={
        'patches_imgs': tf.io.FixedLenFeature([], tf.string),
        'patches_maps': tf.io.FixedLenFeature([], tf.string),
        'tes':          tf.io.FixedLenFeature([], tf.string),
        'sigma':        tf.io.FixedLenFeature([], tf.string),
        'Nx': tf.io.FixedLenFeature([], tf.int64),
        'Ny': tf.io.FixedLenFeature([], tf.int64),
        'Nc': tf.io.FixedLenFeature([], tf.int64),
        'Np': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example,feature_discription)

    Nx = parsed_example['Nx']
    Ny = parsed_example['Ny']
    Nc = parsed_example['Nc']
    Np = parsed_example['Np']
    patches_imgs    = tf.io.parse_tensor(parsed_example['patches_imgs'],out_type=tf.float32)
    patches_maps    = tf.io.parse_tensor(parsed_example['patches_maps'],out_type=tf.float32)
    tes             = tf.io.parse_tensor(parsed_example['tes'],  out_type=tf.float32)
    sigma           = tf.io.parse_tensor(parsed_example['sigma'],out_type=tf.float32)

    patches_imgs    = tf.reshape(patches_imgs,shape=[Nx,Ny,Nc])
    patches_maps    = tf.reshape(patches_maps,shape=[Nx,Ny,Np])
    sigma           = tf.reshape(sigma,shape=[Nx,Ny,1])
    tes             = tf.reshape(tes,  shape=[Nc])

    return (patches_imgs,patches_maps,sigma,tes)

def extract(imgs,maps,sigma,tes,rescale=[1.0,1.0],model_type='physic-inform'):
    imgs_rescale  = tf.math.divide(imgs,rescale[0])
    maps_rescale  = tf.math.divide(maps,rescale)
    sigma_rescale = tf.math.divide(sigma,rescale[0])
    tes_rescale   = tf.math.multiply(tes,rescale[1])
    if model_type == 'physic-inform':
        return ((imgs_rescale,sigma_rescale,tes_rescale),(imgs_rescale,maps_rescale))

def get_len(dataset):
    """
    Get the length of dataset.
    ### ARGUMENTS
    - dataset, tf.data.Dataset

    ### RETURN
    - length, the length of dataset.
    """
    return sum(1 for _ in dataset)

def mean_std_roi(x,mask):
    """
    Calculate the mena and the std in ROI of the data (x).
    ### ARGUMENTS
    - x, shape = [batch, h, w]
    - mask, shape = [batch,h,w]

    ### RETURN
    - mean, shape = [batch], the mean value in ROI of each study.
    - std, shape = [batch], the std in ROI of each study.
    """
    mask = (mask+1)%2 # region of maskout (0 -> 1)
    x_masked = np.ma.array(x,mask=mask)
    mean = x_masked.mean(axis=(1,2))
    std  = x_masked.std(axis=(1,2))

    return mean, std

def bland_altman_plot(x,y,xLimit=1000.0,yLimit=200,title='Method'):
    """
    Bland Altman plot (x-y).
    ### ARGUMENTS
    - x, shape = [batch,h,w].
    - y, shape = [batch,h,w].
    - mask, shape = [batch,h,w].

    ### RETURN
    Plot a Bland Altman plot into figure.
    """
    mean = np.mean([x, y], axis=0)
    diff = x - y      # Difference between data1 and data2
    md   = np.mean(diff)        # Mean of the difference
    sd   = np.std(diff, axis=0) # Standard deviation of the difference

    plt.plot(mean, diff,'o',color='blue') # data point
    plt.axhline(md,           color='black', linestyle='-')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    font_size = 15.0
    # plt.title('('+str(np.round(md+1.96*sd,2))+','+str(np.round(md,2))+','+str(np.round(md-1.96*sd,2))+')',loc='right')
    if np.abs(md+1.96*sd) < yLimit:
        plt.text(x=15,y=md + 1.96*sd+10,s='+1.96SD',fontsize=font_size,color='black')
        plt.text(x=15+875,y=md+1.96*sd+10,s='{:>6.2f}'.format(md+1.96*sd),fontsize=font_size,color='red')

    if np.abs(md-1.96*sd) < yLimit:
        plt.text(x=15,y=md - 1.96*sd-20,s='-1.96SD',fontsize=font_size,color='black')
        plt.text(x=15+875,y=md-1.96*sd-20,s='{:>6.2f}'.format(md-1.96*sd),fontsize=font_size,color='red')

    plt.text(x=15+875,y=md+3,s='{:6.2f}'.format(md),fontsize=font_size,color='red')

    plt.ylim([-yLimit,yLimit]),plt.ylabel('{} - Reference ($s^{{-1}}$)'.format(title),fontsize=font_size)
    plt.xlim([0,xLimit]),plt.xlabel('({} + Reference)/2 ($s^{{-1}}$)'.format(title),fontsize=font_size)

if __name__ == '__main__':
    tes  = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    path = os.path.join('data_liver_mr_same_te','Study*')

    data = read_dicom_itk(path)
    print(data.shape)

    # data      = np.load(os.path.join('InVivo','wImg121.npy'))
    maskLiver = np.load(os.path.join('InVivo','maskLiver.npy'))
    maskBody  = np.load(os.path.join('InVivo','maskBody.npy'))
    maskParen = np.load(os.path.join('InVivo','maskParenchyma.npy'))
    print(maskLiver.shape)

    # info = data_info(path)
    patches_imgs        = patch(data=data,mask=maskBody[...,0],patch_size=32,step_size=8)
    patches_mask_liver  = patch(data=maskLiver,mask=maskBody[...,0],patch_size=32,step_size=8)
    patches_mask_body   = patch(data=maskBody,mask=maskBody[...,0],patch_size=32,step_size=8)
    patches_mask_paren  = patch(data=maskParen,mask=maskBody[...,0],patch_size=32,step_size=8)

    patches_imgs_n = addNoise(patches_imgs,sigma=10.0,noise_type='Rician', NCoils=1)

    sigmas = sigma_bkg(data)
    print(sigmas[0:10])

    patches,idx = patch(data=data,mask=maskBody[...,0],patch_size=32,step_size=8,id=True)
    sigmas      = rearrange(sigmas,idx)
    print(sigmas.shape)


    # data = data[0:2]
    # maskParen = maskParen[0:2]
    # # maps = image2map(data,tes,method='ATF',model='Truncation',mask=maskParen[...,0],num_coils=1,fix_sigma=False)
    # maps = image2map(data,tes,method='PW',model='Truncation',mask=maskParen[...,0],num_coils=1,fix_sigma=False)
    # print(maps.shape)

    # for i in range(maps.shape[0]):
    #     # time.sleep(1.0)
    #     plt.figure(figsize=(10,5))
    #     plt.subplot(2,2,1), plt.imshow(data[i,...,0],cmap='gray',vmin=0.0,vmax=500.0),plt.axis('off'),plt.title(i+1),plt.colorbar(fraction=0.022)
    #     plt.subplot(2,2,2), plt.imshow(maps[i,...,0],cmap='jet', vmin=0.0,vmax=500.0),plt.axis('off'),plt.colorbar(fraction=0.022)
    #     plt.subplot(2,2,3), plt.imshow(maps[i,...,1],cmap='jet', vmin=0.0,vmax=500.0),plt.axis('off'),plt.colorbar(fraction=0.022)
    #     plt.subplot(2,2,4), plt.imshow(maskParen[i,...,0],cmap='jet'),plt.axis('off'),plt.colorbar(fraction=0.022)
    #     plt.savefig('figures/tmp')