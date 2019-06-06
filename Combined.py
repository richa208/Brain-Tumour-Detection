import tensorlayer as tl
import numpy as np
import os, csv, random, gc, time, pickle
import nibabel as nib
import tensorflow as tf
from tensorlayer.layers import *
from google.colab import files
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Load the Drive helper and mount
from google.colab import drive as dd
dd.mount('/content/drive')

from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
drive_service = build('drive', 'v3')




DATA_SIZE = 'small'

save_dir = '/data/train_dev_all/'

if not os.path.exists(save_dir):
  os.makedirs(save_dir)
  
HGG_data_path = "/content/drive/My Drive/BRATS2018/HGG"
LGG_data_path = "/content/drive/My Drive/BRATS2018/LGG"
survival_csv_path = "/content/drive/My Drive/BRATS2018/survival_data.csv"

survival_id_list = []
survival_age_list = []
survival_peroid_list = []

with open(survival_csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for idx, content in enumerate(reader):
        survival_id_list.append(content[0])
        survival_age_list.append(float(content[1]))
        survival_peroid_list.append(float(content[2]))

#print(len(survival_id_list)) #163

if DATA_SIZE == 'all':
    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)
    LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)
elif DATA_SIZE == 'half':
    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)[0:100]# DEBUG WITH SMALL DATA
    LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)[0:30] # DEBUG WITH SMALL DATA
elif DATA_SIZE == 'small':
    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)[0:25] # DEBUG WITH SMALL DATA
    LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)[0:10] # DEBUG WITH SMALL DATA
else:
    exit("Unknow DATA_SIZE")
    
print("length of HGG and LGG data list = ",len(HGG_path_list), len(LGG_path_list)) #210 #75

HGG_name_list = [os.path.basename(p) for p in HGG_path_list] #extracting file names
LGG_name_list = [os.path.basename(p) for p in LGG_path_list]

survival_id_from_HGG = []  #used in mean and deviation calculation
survival_id_from_LGG = []
for i in survival_id_list:   #changes are made here for only LGG
    if i in HGG_name_list:
        survival_id_from_HGG.append(i)
    elif i in LGG_name_list:
        survival_id_from_LGG.append(i)
        
print(len(survival_id_from_HGG), len(survival_id_from_LGG)) #163, 0

index_HGG = list(range(0, len(survival_id_from_HGG)))
# index_HGG = []
index_LGG = list(range(0, 10))
# index_LGG = list(range(0, len(survival_id_from_LGG)))

if DATA_SIZE == 'all':
    dev_index_HGG = index_HGG[-84:-42]
    test_index_HGG = index_HGG[-42:]
    tr_index_HGG = index_HGG[:-84]
    dev_index_LGG = index_LGG[-30:-15]
    test_index_LGG = index_LGG[-15:]
    tr_index_LGG = index_LGG[:-30]
elif DATA_SIZE == 'half':
    dev_index_HGG = index_HGG[-30:]  # DEBUG WITH SMALL DATA
    test_index_HGG = index_HGG[-5:]
    tr_index_HGG = index_HGG[:-30]
    dev_index_LGG = index_LGG[-10:]  # DEBUG WITH SMALL DATA
    test_index_LGG = index_LGG[-5:]
    tr_index_LGG = index_LGG[:-10]
elif DATA_SIZE == 'small':
    dev_index_HGG = index_HGG[-5:]   # DEBUG WITH SMALL DATA
    test_index_HGG = index_HGG[-10:-5]
    tr_index_HGG = index_HGG[:-10]
    dev_index_LGG = index_LGG[-5:]    # DEBUG WITH SMALL DATA
    test_index_LGG = index_LGG[-6:-5]
    tr_index_LGG = index_LGG[:-6]
    
    
survival_id_dev_HGG = [survival_id_from_HGG[i] for i in dev_index_HGG]
survival_id_test_HGG = [survival_id_from_HGG[i] for i in test_index_HGG]
survival_id_tr_HGG = [survival_id_from_HGG[i] for i in tr_index_HGG]

survival_id_dev_LGG = [LGG_name_list[i] for i in dev_index_LGG]
survival_id_test_LGG = [LGG_name_list[i] for i in test_index_LGG]
survival_id_tr_LGG = [LGG_name_list[i] for i in tr_index_LGG]

print("survival_id_dev_LGG = ", len(survival_id_dev_LGG), "survival_id_tr_LGG = ",len(survival_id_tr_LGG))


#only of HGG, not of LGG
survival_age_dev = [survival_age_list[survival_id_list.index(i)] for i in survival_id_dev_HGG]
survival_age_test = [survival_age_list[survival_id_list.index(i)] for i in survival_id_test_HGG]
survival_age_tr = [survival_age_list[survival_id_list.index(i)] for i in survival_id_tr_HGG]

survival_period_dev = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_dev_HGG]
survival_period_test = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_test_HGG]
survival_period_tr = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_tr_HGG]

data_types = ['flair', 't2', 't1ce']
data_types_mean_std_dict = {i: {'mean': 0.0, 'std': 1.0} for i in data_types}

#==================== LOAD ALL IMAGES' PATH AND COMPUTE MEAN/ STD
for i in data_types:
    data_temp_list = []
    for j in survival_id_from_HGG:
        img_path = os.path.join(HGG_data_path, j, j + '_' + i + '.nii.gz')
        img = nib.load(img_path).get_data()
        data_temp_list.append(img)

    for j in survival_id_from_LGG:
        img_path = os.path.join(LGG_data_path, j, j + '_' + i + '.nii.gz')
        img = nib.load(img_path).get_data()
        data_temp_list.append(img)

    data_temp_list = np.asarray(data_temp_list)
    m = np.mean(data_temp_list)
    s = np.std(data_temp_list)
    data_types_mean_std_dict[i]['mean'] = m
    data_types_mean_std_dict[i]['std'] = s
del data_temp_list


print(data_types_mean_std_dict)

with open(save_dir + 'mean_std_dict.pickle', 'wb') as f:
    pickle.dump(data_types_mean_std_dict, f, protocol=4)
    
exit()
    
##==================== GET NORMALIZE IMAGES
X_train_input = []
X_train_target = []
# X_train_target_whole = [] # 1 2 4
# X_train_target_core = [] # 1 4
# X_train_target_enhance = [] # 4

X_dev_input = []
X_dev_target = []
# X_dev_target_whole = [] # 1 2 4
# X_dev_target_core = [] # 1 4
# X_dev_target_enhance = [] # 4

print(" HGG Validation")
for i in survival_id_dev_HGG:
    print(i)
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(HGG_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = img.astype(np.float64)
        all_3d_data.append(img)
        
    seg_path = os.path.join(HGG_data_path, i, i + '_seg.nii.gz')
    seg_img = nib.load(seg_path).get_data()
    seg_img = np.transpose(seg_img, (1, 0, 2))
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float64)
        X_dev_input.append(combined_array)

        seg_2d = seg_img[:, :, j]
        seg_2d.astype(int)                          
        
        X_dev_target.append(seg_2d)
    del all_3d_data
    gc.collect()
    # print("finished {}".format(i))

print(" LGG Validation")
for i in survival_id_dev_LGG:
    print(i)
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(LGG_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = img.astype(np.float64)
        all_3d_data.append(img)

    seg_path = os.path.join(LGG_data_path, i, i + '_seg.nii.gz')
    seg_img = nib.load(seg_path).get_data()
    seg_img = np.transpose(seg_img, (1, 0, 2))
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float64)
        X_dev_input.append(combined_array)

        seg_2d = seg_img[:, :, j]
        seg_2d.astype(int)
        X_dev_target.append(seg_2d)
    del all_3d_data
    gc.collect()

X_dev_input = np.asarray(X_dev_input, dtype=np.float64)
X_dev_target = np.asarray(X_dev_target, dtype=np.float64)

# with open(save_dir + 'dev_input.pickle', 'wb') as f:
#     pickle.dump(X_dev_input, f, protocol=4)
# with open(save_dir + 'dev_target.pickle', 'wb') as f:
#     pickle.dump(X_dev_target, f, protocol=4)

# del X_dev_input, X_dev_target

print(" HGG Train")
for i in survival_id_tr_HGG:
    print(i)
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(HGG_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = img.astype(np.float64)
        all_3d_data.append(img)

    seg_path = os.path.join(HGG_data_path, i, i + '_seg.nii.gz')
    seg_img = nib.load(seg_path).get_data()
    seg_img = np.transpose(seg_img, (1, 0, 2))
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float64)
        X_train_input.append(combined_array)

        seg_2d = seg_img[:, :, j]
       
        seg_2d.astype(int)
        X_train_target.append(seg_2d)
    del all_3d_data

print(" LGG Train")
for i in survival_id_tr_LGG:
    print(i)
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(LGG_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = img.astype(np.float64)
        all_3d_data.append(img)

    seg_path = os.path.join(LGG_data_path, i, i + '_seg.nii.gz')
    seg_img = nib.load(seg_path).get_data()
    seg_img = np.transpose(seg_img, (1, 0, 2))
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float64)
        X_train_input.append(combined_array)

        seg_2d = seg_img[:, :, j]
        seg_2d.astype(int)
        X_train_target.append(seg_2d)
    del all_3d_data
    
X_train_input = np.asarray(X_train_input, dtype=np.float64)
X_train_target = np.asarray(X_train_target, dtype=np.float64)


print("completed prepare_data_with_valid")

def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)

def u_net(x, is_train=False, reuse=False, n_out=1):
    _, nx, ny, nz = x.get_shape().as_list()
    print(x.get_shape().as_list())
    with tf.variable_scope("u_net", reuse=reuse) as scope:
       # tl.layers.set_name_reuse(reuse)
        if reuse:
            scope.reuse_variables()
        inputs = InputLayer(x, name='inputs')
        conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', name='conv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', name='conv1_2')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', name='conv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', name='conv2_2')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', name='conv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', name='conv3_2')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', name='conv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', name='conv4_2')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', name='conv5_1')
        conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', name='conv5_2')

        up4 = DeConv2d(conv5, 512, (3, 3), (2, 2), name='deconv4')
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', name='uconv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', name='uconv4_2')
        up3 = DeConv2d(conv4, 256, (3, 3), (2, 2), name='deconv3')
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', name='uconv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', name='uconv3_2')
        up2 = DeConv2d(conv3, 128, (3, 3), (2, 2), name='deconv2')
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', name='uconv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', name='uconv2_2')
        up1 = DeConv2d(conv2, 64, (3, 3), (2, 2), name='deconv1')
        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', name='uconv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', name='uconv1_2')
        conv1 = Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, padding='SAME', name='uconv1')
    return conv1
  
def distort_imgs(data):
    """ data augumentation """
    x1, x3, x4, y = data
    # x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],  # previous without this, hard-dice=83.7
    #                         axis=0, is_random=True) # up down
    x1, x3, x4, y = tl.prepro.flip_axis_multi([x1, x3, x4, y],
                            axis=1, is_random=True) # left right
#     x1, x3, x4, y = tl.prepro.elastic_transform_multi([x1, x3, x4, y],
#                             alpha=720, sigma=24, is_random=True)
    x1, x3, x4, y = tl.prepro.rotation_multi([x1, x3, x4, y], rg=20,
                            is_random=True, fill_mode='constant') # nearest, constant
    x1, x3, x4, y = tl.prepro.shift_multi([x1, x3, x4, y], wrg=0.10,
                            hrg=0.10, is_random=True, fill_mode='constant')  #can try different values for shifting
#     x1, x3, x4, y = tl.prepro.shear_multi([x1, x3, x4, y], 0.05,
#                             is_random=True, fill_mode='constant')
#     x1, x3, x4, y = tl.prepro.zoom_multi([x1, x3, x4, y],
#                             zoom_range=[0.9, 1.1], is_random=True,
#                             fill_mode='constant')
    return x1, x3, x4, y

def vis_imgs(X, y, path):
    """ show one slice """
    if y.ndim == 2:  #.ndim gives the number of dimensions
        y = y[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0],
        X[:,:,1], X[:,:,2], y[:,:,0]]), size=(1, 4),
        image_path=path)  # this gives the warning of conversion of float to uint

def vis_imgs2(X, y_, y, path):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis], y_, y]), size=(1, 5),
        image_path=path)

def main(task='all'):
    ## Create folder to save trained model and result images
    save_dir = "/content/checkpoint"
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir("samples/{}".format(task))

    ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    # you will get X_train_input, X_train_target, X_dev_input and X_dev_target
    # there are 4 labels in targets:
    # Label 0: background
    # Label 1: necrotic and non-enhancing tumor
    # Label 2: edema
    # Label 4: enhancing tumor
    
#     import prepare_data_with_valid as dataset
#     X_train = dataset.X_train_input
#     y_train = dataset.X_train_target[:,:,:,np.newaxis]
#     X_test = dataset.X_dev_input
#     y_test = dataset.X_dev_target[:,:,:,np.newaxis]
    print(X_train_input.shape)
    X_train = X_train_input
    y_train = X_train_target[:,:,:,np.newaxis]
    X_test = X_dev_input
    y_test = X_dev_target[:,:,:,np.newaxis]
    
    print("X_train.shape = ", X_train.shape)
    print("X_train_target.shape = ",X_train_target.shape)
    print("y_train.shape = ",y_train.shape)
    if task == 'all':
        y_train = (y_train > 0).astype(int)
        y_test = (y_test > 0).astype(int)
    elif task == 'necrotic':
        y_train = (y_train == 1).astype(int)
        y_test = (y_test == 1).astype(int)
    elif task == 'edema':
        y_train = (y_train == 2).astype(int)
        y_test = (y_test == 2).astype(int)
    elif task == 'enhance':
        y_train = (y_train == 4).astype(int)
        y_test = (y_test == 4).astype(int)
    elif task == 'core':
        y_train = np.logical_or(y_train==4,y_train==1).astype(int)
        y_test = np.logical_or(y_test==4,y_test==1).astype(int)
    else:
        exit("Unknow task %s" % task)

    ###======================== HYPER-PARAMETERS ============================###
    batch_size = 20
    lr = 0.0001 
    # lr_decay = 0.5
    # decay_every = 100
    beta1 = 0.9
    n_epoch = 30
    print_freq_step = 100

    ###======================== SHOW DATA ===================================###
    # show one slice
    X = np.asarray(X_train[10])
    y = np.asarray(y_train[10])
    print(X.shape, X.min(), X.max()) # (240, 240, 4) -0.380588 2.62761
    print(y.shape, y.min(), y.max()) # (240, 240, 1) 0 1
    nw, nh, nz = X.shape
    vis_imgs(X, y, 'samples/{}/_train_im.png'.format(task))
    # show data augumentation results
    for i in range(10):
        x_flair, x_t2, x_t1ce, label = distort_imgs([X[:,:,0,np.newaxis], X[:,:,1,np.newaxis], X[:,:,2,np.newaxis], y])#[:,:,np.newaxis]])
        X_dis = np.concatenate((x_flair, x_t2, x_t1ce), axis=2)
        vis_imgs(X_dis, label, 'samples/{}/_train_im_aug{}.png'.format(task, i)) 
    
#     try:
#       files.download("_train_im_aug{}.png".format(0))
#       files.download('/content/samples/all/_train_im_aug{}.png'.format(0))
#       uploaded = drive.CreateFile({'title': '_train_im_aug{}_{}_{}_{}.png'.format(0,task,0,0)})
#       uploaded.SetContentFile('/content/samples/{}/_train_im_aug{}.png'.format(task,0))
#       uploaded.Upload()
#       print('Uploaded file with ID {}'.format(uploaded.get('id')))
#       file_metadata = {
#         'name': '_train_{}_im_dummy_aug{}.png'.format(task,0),
#         'mimeType': 'image/png',
#         'parents': ['19aRVWFG6ZZneRQ869u6vYlhAVtjvep7f']
#       }
#       media = MediaFileUpload('/content/samples/{}/_train_im_aug{}.png'.format(task,0), 
#                               mimetype='image/png',
#                               resumable=True)
#       created = drive_service.files().create(body=file_metadata,
#                                              media_body=media,
#                                              fields='id').execute()
#       print('File ID: {} and train_{}_{}_{}.png '.format(created.get('id'),task,0,0))
#     except Exception as e:
#       print("file not found" + str(e))
  
    
    with tf.device('/GPU:0'):
        tf.reset_default_graph()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        with tf.device('/GPU:0'): #<- remove it if you train on CPU or other GPU
            ###======================== DEFIINE MODEL =======================###
            ## nz is 4 as we input all Flair, T1, T1c and T2.
            t_image = tf.placeholder('float32', [batch_size, nw, nh, nz], name='input_image')
            ## labels are either 0 or 1
            t_seg = tf.placeholder('float32', [batch_size, nw, nh, 1], name='target_segment')
            ## train inference
            #net = model.u_net(t_image, is_train=True, reuse=False, n_out=1)
            net = u_net(t_image, is_train=True, reuse=False, n_out=1)
            ## test inference
            #net_test = model.u_net(t_image, is_train=False, reuse=True, n_out=1)
            net_test = u_net(t_image, is_train=False, reuse=True, n_out=1)

            ###======================== DEFINE LOSS =========================###
            ## train losses
            out_seg = net.outputs
            dice_loss = 1 - tl.cost.dice_coe(out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
            iou_loss = tl.cost.iou_coe(out_seg, t_seg, axis=[0,1,2,3])
            dice_hard = tl.cost.dice_hard_coe(out_seg, t_seg, axis=[0,1,2,3])
            loss = dice_loss

            ## test losses
            test_out_seg = net_test.outputs
            test_dice_loss = 1 - tl.cost.dice_coe(test_out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
            test_iou_loss = tl.cost.iou_coe(test_out_seg, t_seg, axis=[0,1,2,3])
            test_dice_hard = tl.cost.dice_hard_coe(test_out_seg, t_seg, axis=[0,1,2,3])

        ###======================== DEFINE TRAIN OPTS =======================###
        t_vars = tl.layers.get_variables_with_name('u_net', True, True)
        with tf.device('/GPU:0'):
            with tf.variable_scope('learning_rate'):
                lr_v = tf.Variable(lr, trainable=True)                               #changed here
            train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=t_vars)

        ###======================== LOAD MODEL ==============================###
        sess.run(tf.global_variables_initializer())                     
        #tl.layers.initialize_global_variables(sess)
        ## load existing model if possible
        tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/u_net_{}.npz'.format(task), network=net)
  
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(shape)
            print(len(shape))
            variable_parameters = 1
            for dim in shape:
                print(dim)
                variable_parameters *= dim.value
            print("variable_parameters =", variable_parameters)
            total_parameters += variable_parameters
        print("total_parameters = ",total_parameters)
        ###======================== TRAINING ================================###
    for epoch in range(0, n_epoch+1):
        epoch_time = time.time()
        print("epoch = {} and epoch_time =  {}".format(epoch, epoch_time))
          
        step_count=0
        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        for batch in tl.iterate.minibatches(inputs=X_train, targets=y_train,
                                    batch_size=batch_size, shuffle=True):
            images, labels = batch
            step_time = time.time()
            print("step_time = {}".format(step_time))
            ## data augumentation for a batch of Flair, T1, T1c, T2 images
            # and label maps synchronously.
            data = tl.prepro.threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis],
                    images[:,:,:,1, np.newaxis], images[:,:,:,2, np.newaxis], labels)],
                    fn=distort_imgs) # (10, 5, 240, 240, 1)
            step_count+=1
            print("step_count = ", step_count)
            b_images = data[:,0:3,:,:,:]  # (10, 4, 240, 240, 1)
            b_labels = data[:,3,:,:,:]
            b_images = b_images.transpose((0,2,3,1,4))
            b_images.shape = (batch_size, nw, nh, nz)

            ## update network
            _, _dice, _iou, _diceh, out = sess.run([train_op,
                    dice_loss, iou_loss, dice_hard, net.outputs],
                    {t_image: b_images, t_seg: b_labels})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

            if n_batch % print_freq_step == 0:
                print("Epoch %d step %d 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)"
                % (epoch, n_batch, _dice, _diceh, _iou, time.time()-step_time))


        print(" ** Epoch [%d/%d] train 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)" %
                (epoch, n_epoch, total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch, time.time()-epoch_time))

        ## save a predition of training set
        for i in range(batch_size):
            if np.max(b_images[i]) > 0:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}_{}.png".format(task, epoch, i))
                file_metadata = {
                  'name': 'train_{}_{}_{}.png'.format(task,epoch,i),
                  'mimeType': 'image/png',
                  'parents': ['1SMzd_fp3-H_aSN4zU_WCXGd2Psl62Xrd']
                }
                media = MediaFileUpload('/content/samples/{}/train_{}_{}.png'.format(task,epoch,i), 
                                        mimetype='image/png',
                                        resumable=True)
                created = drive_service.files().create(body=file_metadata,
                                                       media_body=media,
                                                       fields='id').execute()
                print('File ID: {} and train_{}_{}_{}.png '.format(created.get('id'),task,epoch,i))
            elif i == batch_size-1:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}_{}.png".format(task, epoch, i))
                file_metadata = {
                  'name': 'train_{}_{}_{}.png'.format(task,epoch,i),
                  'mimeType': 'image/png',
                  'parents': ['1SMzd_fp3-H_aSN4zU_WCXGd2Psl62Xrd']
                }
                media = MediaFileUpload('/content/samples/{}/train_{}_{}.png'.format(task,epoch,i), 
                                        mimetype='image/png',
                                        resumable=True)
                created = drive_service.files().create(body=file_metadata,
                                                       media_body=media,
                                                       fields='id').execute()
                print('File ID: {} and train_{}_{}_{}.png '.format(created.get('id'),task,epoch,i))

        ###======================== EVALUATION ==========================###
        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test,
                                        batch_size=batch_size, shuffle=True):  #shuffle=True
            b_images, b_labels = batch
            _dice, _iou, _diceh, out = sess.run([test_dice_loss,
                    test_iou_loss, test_dice_hard, net_test.outputs],
                    {t_image: b_images, t_seg: b_labels})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

        print(" **"+" "*17+"test 1-dice: %f hard-dice: %f iou: %f (2d no distortion)" %
                (total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch))
        print(" task: {}".format(task))
        ## save a predition of test set
        for i in range(batch_size):
            if np.max(b_images[i]) > 0:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/test_{}_{}.png".format(task, epoch, i))
                file_metadata = {
                  'name': 'test_{}_{}_{}.png'.format(task,epoch,i),
                  'mimeType': 'image/png',
                  'parents': ['1SMzd_fp3-H_aSN4zU_WCXGd2Psl62Xrd']
                }
                media = MediaFileUpload('/content/samples/{}/test_{}_{}.png'.format(task,epoch,i), 
                                        mimetype='image/png',
                                        resumable=True)
                created = drive_service.files().create(body=file_metadata,
                                                       media_body=media,
                                                       fields='id').execute()
                print('File ID: {} and test_{}_{}_{}.png '.format(created.get('id'),task,epoch,i))
            elif i == batch_size-1:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/test_{}_{}.png".format(task, epoch, i))
                file_metadata = {
                  'name': 'test_{}_{}_{}.png'.format(task,epoch,i),
                  'mimeType': 'image/png',
                  'parents': ['1SMzd_fp3-H_aSN4zU_WCXGd2Psl62Xrd']
                }
                media = MediaFileUpload('/content/samples/{}/test_{}_{}.png'.format(task,epoch,i), 
                                        mimetype='image/png',
                                        resumable=True)
                created = drive_service.files().create(body=file_metadata,
                                                       media_body=media,
                                                       fields='id').execute()
                print('File ID: {} and test_{}_{}_{}.png '.format(created.get('id'),task,epoch,i))

        ###======================== SAVE MODEL ==========================###
        tl.files.save_npz(net.all_params, name=save_dir+'/u_net_{}_{}.npz'.format(task, epoch), sess=sess)
     

if __name__ == "__main__":

    main('core')
  



