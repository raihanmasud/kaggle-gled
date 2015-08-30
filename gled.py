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


###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:
    y_raw = []
    raw = []
    ##debug code remove
    if subject > 1:
        continue

    # ################ READ DATA ################################################
    fnames = glob('../train1/subj%d_series*_data.csv' % (subject))

    for fname in fnames:
        data, labels = prepare_data_train(fname)
        raw.append(data)
        y_raw.append(labels)

    X = pd.concat(raw)
    y = pd.concat(y_raw)

    # transform in numpy array
    # transform train data in numpy array
    X_train = np.asarray(X.astype(np.float32))
    y = np.asarray(y.astype(np.float32))

################ Read test data #####################################

    fnames = glob('../test1/subj%d_series*_data.csv' % (subject))

    test = []
    idx = []
    for fname in fnames:
        data = prepare_data_test(fname)
        test.append(data)
        idx.append(np.array(data['id']))

    X_test = pd.concat(test)
    ids = np.concatenate(idx)
    ids_tot.append(ids)
    X_test = X_test.drop(['id'], axis=1)  # remove id
    # transform test data in numpy array
    X_test = np.asarray(X_test.astype(np.float32))


####process training data####
X = X_train
X = data_preprocess_train(X)
NO_TIME_POINTS = 100
TOTAL_TIME_POINTS = len(X) // NO_TIME_POINTS

no_rows = TOTAL_TIME_POINTS * NO_TIME_POINTS
X = X[0:no_rows, :]
print('X ', X.shape)

X = X.transpose()
X_Samples = np.split(X, TOTAL_TIME_POINTS, axis=1)
X = np.asarray(X_Samples)
print('X({0})'.format(X.shape))

y = y[0:no_rows, :]
y = y[::NO_TIME_POINTS, :]
print('y({0})'.format(y.shape))

####process test data####
X_test = X_test
X_test = data_preprocess_test(X_test)
NO_TIME_POINTS = 100
TOTAL_TIME_POINTS = len(X_test) // NO_TIME_POINTS
REMAINDER_TEST_POINTS = len(X_test) % NO_TIME_POINTS

no_rows = TOTAL_TIME_POINTS * NO_TIME_POINTS
X_test = X_test[0:no_rows, :]
print('X_test ', X_test.shape)

X_test = X_test.transpose()
X_test_Samples = np.split(X_test, TOTAL_TIME_POINTS, axis=1)
X_test = np.asarray(X_test_Samples)
print('X_test({0})'.format(X_test.shape))


#ids_tot[0] = ids_tot[0][:no_rows]
#ids_tot[0] = ids_tot[0][::NO_TIME_POINTS]
#print('id_test {}'.format(len(ids_tot[0])))

###########################################################################

def float32(k):
    return np.cast['float32'](k)


channels = 32  # no. of input
batch_size = None  #None = arbitary batch size
hidden_layer_size = 100  #change to 1024
N_EVENTS = 6
max_epochs = 5  #increase it


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

    batch_iterator_train = BatchIterator(batch_size=20),
    batch_iterator_test = BatchIterator(batch_size=20),

    y_tensor_type=theano.tensor.matrix,
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    objective_loss_function=loss,
    regression=True,

    max_epochs=max_epochs,
    verbose=1,
)

net.fit(X, y)

params = net.get_all_params_values()
learned_weights = net.load_params_from(params)
probabilities = net.predict_proba(X_test)

for i, p in enumerate(probabilities):
    print("subj{0}_series{1}_{2}: {3}".format(1,9,i, p)) #Todo: update for all subjects
    for j in range(NO_TIME_POINTS):
        pred_tot.append(p)

for k in range(REMAINDER_TEST_POINTS):
    pred_tot.append(pred_tot[-1])


# submission file
submission_file = 'gled_conv_net_grasp.csv'
# # create pandas object for sbmission

submission = pd.DataFrame(index=np.concatenate(ids_tot),
                           columns=cols,
                           data=pred_tot)
# # write file
submission.to_csv(submission_file, index_label='id', float_format='%.6f')
# submission file
















