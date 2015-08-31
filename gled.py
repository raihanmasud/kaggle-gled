__author__ = 'raihan'

"""
Borrowing from Elena Cuoco's data loading... &
ConvNet Model from Denial Nouri's kfkd and Tim Hochberg's script
"""

import numpy as np
import pandas as pd
from glob import glob
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
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
    # normalize data
    mean = X.mean(axis=0)
    X -= mean
    std = X.std(axis=0)
    X /= std
    X_prep = X
    return X_prep


def data_preprocess_test(X):
    # normalizing data
    mean = X.mean(axis=0)
    X -= mean
    std = X.std(axis=0)
    X /= std
    X_prep = X
    return X_prep

#######columns name for labels#############
cols = ['HandStart', 'FirstDigitTouch',
        'BothStartLoadPhase', 'LiftOff',
        'Replace', 'BothReleased']

#######number of subjects###############
subjects = range(1, 13)
ids_tot = []
pred_tot = []
test_dict = dict()

def float32(k):
    return np.cast['float32'](k)


channels = 32  # no. of input
batch_size = None  #None = arbitary batch size
hidden_layer_size = 100  #change to 1024
N_EVENTS = 6
max_epochs = 5  #increase it
NO_TIME_POINTS = 100

test_total = 0

def loss(x, t):
    return aggregate(binary_crossentropy(x, t))

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv1', layers.Conv1DLayer),
        ('conv2', layers.Conv1DLayer),
        ('pool1', layers.MaxPool1DLayer),
        ('dropout2', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('output', layers.DenseLayer),
    ],
    input_shape=(None, channels, NO_TIME_POINTS),
    dropout1_p=0.5,
    conv1_num_filters=4, conv1_filter_size=1,
    conv2_num_filters=8, conv2_filter_size=4, pool1_pool_size=4,
    dropout2_p=0.5, hidden4_num_units=hidden_layer_size,
    dropout3_p=0.5, hidden5_num_units=hidden_layer_size,
    dropout4_p=0.5, output_num_units=N_EVENTS, output_nonlinearity=sigmoid,

    batch_iterator_train = BatchIterator(batch_size=1000),
    batch_iterator_test = BatchIterator(batch_size=1000),

    y_tensor_type=theano.tensor.matrix,
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    objective_loss_function=loss,
    regression=True,

    max_epochs=max_epochs,
    verbose=1,
)



###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:
    y_raw = []
    raw = []
    ##debug code remove
    #if subject > 1:
    #    continue

    # ################ READ DATA ################################################
    fnames = glob('../train1/subj%d_series*_data.csv' % (subject))

    for fname in fnames:
        data, labels = prepare_data_train(fname)
        raw.append(data)
        y_raw.append(labels)

    if raw and y_raw:
        X = pd.concat(raw)
        y = pd.concat(y_raw)

    # transform in numpy array
    # transform train data in numpy array
        X_train = np.asarray(X.astype(np.float32))
        y_train = np.asarray(y.astype(np.float32))

####process training data####
    X = X_train
    X = data_preprocess_train(X)
    total_time_points = len(X) // NO_TIME_POINTS

    no_rows = total_time_points * NO_TIME_POINTS
    X = X[0:no_rows, :]
    print('X ', X.shape)

    X = X.transpose()
    X_Samples = np.split(X, total_time_points, axis=1)
    X = np.asarray(X_Samples)
    print('X({0})'.format(X.shape))

    y = y_train
    y = y[0:no_rows, :]
    y = y[::NO_TIME_POINTS, :]
    print('y({0})'.format(y.shape))

    print("Training subject%d...." %(subject))
    net.fit(X,y)

################ Read test data #####################################

    fnames = glob('../test/subj%d_series*_data.csv' % (subject))

    test = []
    idx = []

    fnames.reverse()
    for fname in fnames:
        data = prepare_data_test(fname)
        test.append(data)
        idx.append(np.array(data['id']))

        data_size = len(data)
        series = 9 if 'series9' in fname else 10
        data_name = 'subj{0}_series{1}'.format(subject, series)
        test_dict[data_name] = data_size

        test_total += data_size
        print('subj{0} test_total= {1}'.format(subject,test_total))

    if idx and test:
        X_test = pd.concat(test)
        ids = np.concatenate(idx)
        ids_tot.append(ids)
        X_test = X_test.drop(['id'], axis=1)  # remove id
    # transform test data in numpy array
    X_test = np.asarray(X_test.astype(np.float32))


####process test data####
    X_test = X_test
    X_test = data_preprocess_test(X_test)
    total_test_time_points = len(X_test) // NO_TIME_POINTS
    remainder_test_points = len(X_test) % NO_TIME_POINTS

    no_rows = total_test_time_points * NO_TIME_POINTS
    X_test = X_test[0:no_rows, :]
    print('X_test ', X_test.shape)

    X_test = X_test.transpose()
    X_test_Samples = np.split(X_test, total_test_time_points, axis=1)
    X_test = np.asarray(X_test_Samples)
    print('X_test({0})'.format(X_test.shape))


###########################################################################

    params = net.get_all_params_values()
    learned_weights = net.load_params_from(params)
    probabilities = net.predict_proba(X_test)

    sub9 = 'subj{0}_series{1}'.format(subject, 9)
    print('sub_ser9 ',sub9)
    data_len9 = test_dict[sub9]
    print('data_len9 ',data_len9)
    total_time_points9 = data_len9 // NO_TIME_POINTS
    print('total_time_points9 ',total_time_points9)
    remainder_data9 = data_len9 % NO_TIME_POINTS
    print('remainder_data9 ',remainder_data9)

    sub10 = 'subj{0}_series{1}'.format(subject, 10)
    print('sub_ser10 ',sub10)
    data_len10 = test_dict[sub10]
    print('data_len10 ',data_len10)
    total_time_points10 = data_len10 // NO_TIME_POINTS
    print('total_time_points10 ',total_time_points10)
    remainder_data10 = data_len10 % NO_TIME_POINTS
    print('remainder_data10 ',remainder_data10)

    data_len_s9_s10_rem10 = data_len9+data_len10-remainder_data10
    print('data_len_s9_s10_rem10 ',data_len_s9_s10_rem10)

    print('len-probab: ', len(probabilities))
    
    for i, p in enumerate(probabilities):
         #or i != data_len_s9_s10_rem10:
        for j in range(NO_TIME_POINTS):
            pred_tot.append(p)
        if i != total_time_points9 :
            print('len-pred_tot',len(pred_tot))
            print('i ',i)
            for k in range(remainder_data9):
                pred_tot.append(pred_tot[-1])
            print('len-pred_tot',len(pred_tot))

    print('len-pred_tot',len(pred_tot))
    for k in range(remainder_data10):
        pred_tot.append(pred_tot[-1])
    print('len-pred_tot',len(pred_tot))

# submission file
submission_file = './gled_conv_net_grasp.csv'
# # create pandas object for sbmission

submission = pd.DataFrame(index=np.concatenate(ids_tot),
                           columns=cols,
                           data=pred_tot)
# # write file
submission.to_csv(submission_file, index_label='id', float_format='%.6f')
# submission file
