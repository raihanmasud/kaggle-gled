__author__ = 'raihan'

"""
Using Elena Cuoco's data loading...
Borrowing idea from Denial Nouri's kfkd and  Tim Hochberg's script
"""

import numpy as np
import pandas as pd
from glob import glob
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator, TrainSplit
import theano
from theano.tensor.nnet import sigmoid
from sklearn.preprocessing import StandardScaler
from lasagne.objectives import aggregate, binary_crossentropy

#############function to read data###########


def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data', '_events')
    # read event file
    labels = pd.read_csv(events_fname)
    clean = data.drop(['id'], axis=1)  # remove id
    labels = labels.drop(['id'], axis=1)  # remove id
    return clean, labels


def prepare_data_test(fname):
    """ read and prepare test data """
    # Read data
    data = pd.read_csv(fname)
    return data


scaler = StandardScaler()


def data_preprocess_train(X):
    #X_prep = scaler.fit_transform(X)
    # do here your preprocessing
    X_prep = X
    return X_prep


def data_preprocess_test(X):
    #X_prep = scaler.transform(X)
    # do here your preprocessing
    X_prep = X
    return X_prep

##downsamplig naive like this is not correct, if you do not low pass filter.
##this down sampling here it needed only to keep the script run below 10 minutes.
## please do not downsample or use correct procedure to decimate data without alias
subsample = 100  # training subsample.if you want to downsample the training data
#######columns name for labels#############
cols = ['HandStart', 'FirstDigitTouch',
        'BothStartLoadPhase', 'LiftOff',
        'Replace', 'BothReleased']




#######number of subjects###############
subjects = range(1, 13)
ids_tot = []
pred_tot = []


###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:
    y_raw = []
    raw = []
    ##debug code remove
    if subject > 1 :
         continue

    # ################ READ DATA ################################################
    #fnames = glob('../train1/subj%d_series*_data_small.csv' % (subject))
    fnames = glob('../train1/subj%d_series*_data.csv' % (subject))

    for fname in fnames:
        data, labels = prepare_data_train(fname)
        raw.append(data)
        y_raw.append(labels)

    X = pd.concat(raw)
    y = pd.concat(y_raw)

    # transform in numpy array
    #transform train data in numpy array
    X_train = np.asarray(X.astype(np.float32))
    #print('X_train ', X_train.shape)
    y = np.asarray(y.astype(np.float32))
    #print('y ',y.shape)

X = X_train
# .reshape( X_train.shape[0],X_train.shape[1])



NO_TIME_POINTS = 100
TOTAL_TIME_POINTS = len(X)// NO_TIME_POINTS

no_rows = TOTAL_TIME_POINTS*NO_TIME_POINTS
X = X[0:no_rows,:]
print('X ',X.shape)


X = X.transpose()
X_Samples = np.split(X,TOTAL_TIME_POINTS, axis=1)
X = np.asarray(X_Samples)
print('X({0})'.format(X.shape))

y = y[0:no_rows,:]
y = y[::NO_TIME_POINTS,:]
print('y({0})'.format(y.shape))



################ Read test data #####################################
#   #Todo: remove after debguggling
    #subject = 9
    #
    # fnames = glob('../test1/subj%d_series*_data_small.csv' % (subject))
    # test = []
    # idx = []
    # for fname in fnames:
    #     data = prepare_data_test(fname)
    #     test.append(data)
    #     idx.append(np.array(data['id']))
    # X_test = pd.concat(test)
    # ids = np.concatenate(idx)
    # ids_tot.append(ids)
    # X_test = X_test.drop(['id'], axis=1)  # remove id
    # # transform test data in numpy array
    # X_test = np.asarray(X_test.astype(float))



# prediction = lasagne.layers.get_output(output_layer)
# loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
# loss = loss.mean()

###########################################################################

def float32(k):
    return np.cast['float32'](k)


channels = 32  # no. of input
batch_size = None #None = arbitary batch size
sample_size = 100 #change to 1024
hidden_layer_size = 100 #change to 1024
N_EVENTS = 6
max_epochs = 10 #increase it

def loss(x,t):
        return aggregate(binary_crossentropy(x, t))


net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        #add Dropout layer beween every two layers; see commented code above
        ('conv1', layers.Conv1DLayer),
        ('conv2', layers.Conv1DLayer),
        ('pool1', layers.MaxPool1DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
    ],
    input_shape=(None, channels, sample_size),
    conv1_num_filters=4, conv1_filter_size=1,
    conv2_num_filters=8, conv2_filter_size=4, pool1_pool_size=4,
    hidden4_num_units=hidden_layer_size, hidden5_num_units=hidden_layer_size,
    output_num_units=N_EVENTS, output_nonlinearity=sigmoid,

    batch_iterator_train = BatchIterator(batch_size=20),

    y_tensor_type=theano.tensor.matrix,  #something not used in kfkd
    update=nesterov_momentum,
    #update_learning_rate=0.01,
    #update_momentum=0.9,

    #optional optimization from kfkd
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    objective_loss_function = loss,
    regression=True,

    #train_split=TrainSplit(eval_size=0.0, stratify=True),

    max_epochs=max_epochs,
    verbose=1,
)

# batch_iterator_train = batch_iter_train,
# batch_iterator_test = batch_iter_test,
# objective_loss_function = loss,


net.fit(X, y)  # submission file






# submission_file = 'grasp-sub-simple.csv'
# # create pandas object for sbmission
# submission = pd.DataFrame(index=np.concatenate(ids_tot),
#                           columns=cols,
#                           data=np.concatenate(pred_tot))
#
# # write file
# submission.to_csv(submission_file, index_label='id', float_format='%.3f')



















# submission
#submission_file = "grasp-sub-simple.csv"
#submission_data = pd.read_csv("C:/Work/kaggle/ecg_grasp_lift/train/sample_submission.csv/sample_submission.csv")
#submission_data = pd.read_csv("C:/Work/kaggle/ecg_grasp_lift/train/train/subj1_series1_events.csv")

#print(submission_data[:2])

#num_data = len(submission_data['id'])
#index_set = set()

# for index, row in submission_data.iterrows():
#     current_index = np.random.randint(0, num_data)
#     if current_index not in index_set:
#         index_set.add(current_index)
#         index += 150
#     for i in range(index, index+150):
#         if(submission_data['HandStart'][i]) != 1:
#             submission_data['HandStart'][i] = 1


#print(submission_data['HandStart'][:20])

#submission
