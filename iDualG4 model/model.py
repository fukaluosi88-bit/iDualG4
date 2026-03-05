from tensorflow.keras.layers import Add, Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

import tensorflow as tf
import numpy as np
import csv

from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Conv2D, Conv1D, Activation, MaxPooling2D, MaxPooling1D, AveragePooling1D, Dropout, \
    Flatten, \
    Dense, BatchNormalization, Input, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Multiply, add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from tensorflow.keras.layers import concatenate, Reshape, Flatten, BatchNormalization, MaxPooling1D, Flatten, LSTM, \
    Dropout, \
    Bidirectional
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import ZeroPadding2D
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Concatenate, Dense, Flatten, GlobalAveragePooling2D, Reshape, Multiply, Add


from tensorflow.keras.layers import Add, Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import numpy as np
import csv
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.layers import Conv2D, Conv1D, Activation, MaxPooling2D, MaxPooling1D, AveragePooling1D, Dropout, \
    Flatten, \
    Dense, BatchNormalization, Input, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Multiply, add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from tensorflow.keras.layers import concatenate, Reshape, Flatten, BatchNormalization, MaxPooling1D, Flatten, LSTM, \
    Dropout, \
    Bidirectional
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import ZeroPadding2D
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.set_printoptions(threshold=np.inf)
np.random.seed(1234)
tf.random.set_seed(42)





def squeeze_excitation(input_tensor, ratio=16):
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling1D()(input_tensor)
    se = Reshape((1, channels))(se)
    se = Dense(int(channels // ratio), activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return Multiply()([input_tensor, se])




def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    eps = 1.0e-4
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    x = BatchNormalization(epsilon=eps, name=conv_name_base + '_x1_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Conv2D(nb_filter, (1, 1), padding='same', use_bias=False, name=conv_name_base + '_x1')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = BatchNormalization(epsilon=eps, name=conv_name_base + '_x2_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = Conv2D(nb_filter, (1, 3), padding='same', use_bias=False, name=conv_name_base + '_x2')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4,
                grow_nb_filters=False):
    eps = 1.0e-4
    concat_feat = x

    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = Concatenate(axis=-1, name='concat_' + str(stage) + '_' + str(branch))([concat_feat, x])
        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter
def emodel(model_input):
    conv1_params = {'nb_filter': 128, 'filter_size': (4, 4), 'pool_size': (2, 2)}
    conv2_params = {'nb_filter': 128, 'filter_size': (4, 4), 'pool_size': (2, 2)}
    conv3_params = {'nb_filter': 128, 'filter_size': (4, 4), 'pool_size': (2, 2)}

    model_input_reshaped = Reshape((2048, 4, 1))(model_input)

    x = Conv2D(filters=conv1_params['nb_filter'], kernel_size=conv1_params['filter_size'], strides=(1, 1),
               activation='relu', padding='same', name='conv1')(model_input_reshaped)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=conv1_params['pool_size'], padding='same', name='pooling1')(x)
    x = Dropout(0.3)(x)

    x, nb_filter = dense_block(x, stage=1, nb_layers=5, nb_filter=64, growth_rate=32, dropout_rate=0.2,
                               weight_decay=1e-4)

    x = Conv2D(filters=conv2_params['nb_filter'], kernel_size=conv2_params['filter_size'], strides=(1, 1),
               activation='relu', padding='same', name='conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=conv2_params['pool_size'], padding='same', name='pooling2')(x)
    x = Dropout(0.3)(x)

    x, nb_filter = dense_block(x, stage=2, nb_layers=5, nb_filter=64, growth_rate=32, dropout_rate=0.2,
                               weight_decay=1e-4)

    x = Conv2D(filters=conv3_params['nb_filter'], kernel_size=conv3_params['filter_size'], strides=(1, 1),
               activation='relu', padding='same', name='conv3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=conv3_params['pool_size'], padding='same', name='pooling3')(x)
    x = Dropout(0.3)(x)

    x, nb_filter = dense_block(x, stage=3, nb_layers=5, nb_filter=64, growth_rate=32, dropout_rate=0.2,
                               weight_decay=1e-4)

    print("emodel_1:", x.shape)

    x = Flatten()(x)
    print("emodel_2:", x.shape)
    x = Dense(256, activation='relu', use_bias=False)(x)
    print("emodel_3:", x.shape)
    x = BatchNormalization(axis=-1)(x)
    print("emodel_4:", x.shape)
    x = Dropout(0.5)(x)
    print("emodel_5:", x.shape)
    model_output = Dense(units=64, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    print("emodel_6:", model_output.shape)

    return model_output

def gmodel(model_input):
    x = Conv1D(filters=256, kernel_size=5, kernel_initializer="he_uniform", padding="same", use_bias=False)(model_input)
    x = Conv1D(filters=256, kernel_size=5, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)

    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = squeeze_excitation(x)

    x = MaxPooling1D(pool_size=2)(x)

    skip_connection1 = x

    x = Conv1D(filters=256, kernel_size=5, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = squeeze_excitation(x)

    x = Add()([x, skip_connection1])
    x = MaxPooling1D(pool_size=2)(x)

    skip_connection2 = x

    x = Conv1D(filters=256, kernel_size=5, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = squeeze_excitation(x)

    x = Add()([x, skip_connection2])
    x = MaxPooling1D(pool_size=2)(x)
    print("gmodel_1:", x.shape)

    x = Flatten()(x)
    print("gmodel_2:", x.shape)
    x = Dense(256, activation='relu', use_bias=False)(x)
    print("gmodel_3:", x.shape)
    x = BatchNormalization(axis=-1)(x)
    print("gmodel_4:", x.shape)
    x = Dense(128, activation='relu', use_bias=False)(x)
    print("gmodel_5:", x.shape)
    x = BatchNormalization(axis=-1)(x)
    print("gmodel_6:", x.shape)
    x = Dropout(0.3)(x)
    print("gmodel_7:", x.shape)
    model_output = Dense(units=64, activation='relu')(x)
    print("gmodel_8:", model_output.shape)

    return model_output


def reverse_complement(seq: str) -> str:

    reverse_seq = seq[::-1].upper()
    complement_list = []
    for i in range(len(reverse_seq)):
        if reverse_seq[i] == 'A':
            complement_list.append('T')
        elif reverse_seq[i] == 'C':
            complement_list.append('G')
        elif reverse_seq[i] == 'G':
            complement_list.append('C')
        elif reverse_seq[i] == 'T':
            complement_list.append('A')
        elif reverse_seq[i] == 'N':
            complement_list.append('N')

    return ''.join(complement_list)

def read_fasta(input_file, reverse=False):
    sequences = []
    current_sequence = ""

    with open(input_file, 'r') as file:
        for line in tqdm(file):
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:

                    if reverse:
                        current_sequence = reverse_complement(current_sequence)
                    sequences.append(current_sequence.upper())
                    current_sequence = ""
                continue
            else:
                current_sequence += line


        if current_sequence:
            if reverse:
                current_sequence = reverse_complement(current_sequence)
            sequences.append(current_sequence.upper())

    return sequences



def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: any = 0,
                   dtype=np.float32) -> np.ndarray:
    """One-hot encode sequence."""

    def to_uint8(string):
        return np.frombuffer(string.encode('ascii'), dtype=np.uint8)

    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)

    return hash_table[to_uint8(sequence)]

def sequnences_to_one_hot(filename, sequences):
    one_hot_datas = []
    for sequence in tqdm(sequences):
        one_hot_data = one_hot_encode(sequence)

        one_hot_datas.append(one_hot_data)

    one_hot_datas = np.array(one_hot_datas)

    return one_hot_datas




positive_path = '/pos_grich_393216_2048bp.fa'
negative_plus_path = '/neg_plus_393216_2048.fa'
negative_minus_path = '/neg_minus_393216_2048.fa'

pos_data = read_fasta(positive_path)


neg_plus_data = read_fasta(negative_plus_path)
neg_minus_data = read_fasta(negative_minus_path, reverse = True)

pos_plus_OneHot = sequnences_to_one_hot("pos_data", pos_data)


neg_plus_OneHot = sequnences_to_one_hot("neg_plus", neg_plus_data)
neg_minus_OneHot = sequnences_to_one_hot("neg_minus", neg_minus_data)
neg_OneHot = np.vstack((neg_plus_OneHot, neg_minus_OneHot))

X_onehot = np.vstack((pos_plus_OneHot, neg_OneHot))



y = np.hstack((np.ones(len(pos_plus_OneHot)), np.zeros(len(neg_OneHot))))


pos_enformer = np.load("/pos_enformer.npy")

neg_plus_enformer = np.load("/neg_plus_enformer.npy")
neg_minus_enformer = np.load("/neg_minus_enformer.npy")

pos_enformer = np.squeeze(pos_enformer)
neg_plus_enformer = np.squeeze(neg_plus_enformer)
neg_minus_enformer = np.squeeze(neg_minus_enformer)


neg_enformer = np.vstack((neg_plus_enformer, neg_minus_enformer))

X_enformer = np.vstack((pos_enformer, neg_enformer))



kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
precision_list = []
recall_list = []
f1_list = []
accuracy_list = []
auc_roc_list = []
aupr_list = []

for fold, (train_all_idx, test_idx) in enumerate(kfold.split(X_onehot, y)):
    print(f"Fold {fold + 1}/{kfold.n_splits}")


    train_idx, val_idx = train_test_split(
        train_all_idx,
        test_size=0.2,
        stratify=y[train_all_idx],
        random_state=42
    )


    X_train_onehot, X_val_onehot, X_test_onehot = X_onehot[train_idx], X_onehot[val_idx], X_onehot[test_idx]

    X_train_enformer = X_enformer[train_idx]
    X_val_enformer = X_enformer[val_idx]
    X_test_enformer = X_enformer[test_idx]

    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]


    emodel_input = Input(shape=X_train_onehot.shape[1:])
    enformer_model_input = Input(shape=X_train_enformer.shape[1:])

    emodel_output = emodel(emodel_input)
    enformer_model_output = gmodel(enformer_model_input)

    merged_output = Concatenate()([emodel_output, enformer_model_output])

    x = Dense(128, activation='relu')(merged_output)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(units=1, activation='sigmoid')(x)
    model = Model(inputs=[emodel_input, enformer_model_input], outputs=output)


    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    checkpoint = ModelCheckpoint(f'/best_model_fold_{fold}.h5',
                                 save_best_only=True, monitor='val_loss')
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)


    model.fit([X_train_onehot, X_train_enformer], y_train,
              validation_data=([X_val_onehot, X_val_enformer], y_val),
              epochs=100,
              batch_size=32,
              callbacks=[checkpoint, early_stopping, reduce_lr])


    model.load_weights(f'/best_model_fold_{fold}.h5')
    y_pred = model.predict([X_test_onehot, X_test_enformer])
    y_pred_class = (y_pred > 0.5).astype(int)


    precision = precision_score(y_test, y_pred_class)
    recall = recall_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)
    accuracy = accuracy_score(y_test, y_pred_class)
    auc_roc = roc_auc_score(y_test, y_pred)
    aupr = average_precision_score(y_test, y_pred)


    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    accuracy_list.append(accuracy)
    auc_roc_list.append(auc_roc)
    aupr_list.append(aupr)

    with open(f'/metrics_fold_{fold}.txt', 'w') as f:
        f.write(f"Test Precision: {precision}\n")
        f.write(f"Test Recall: {recall}\n")
        f.write(f"Test F1 Score: {f1}\n")
        f.write(f"Test Accuracy: {accuracy}\n")
        f.write(f"Test AUC-ROC: {auc_roc}\n")
        f.write(f"Test AUPR: {aupr}\n")


precision_mean = np.mean(precision_list)
precision_std = np.std(precision_list)
recall_mean = np.mean(recall_list)
recall_std = np.std(recall_list)
f1_mean = np.mean(f1_list)
f1_std = np.std(f1_list)
accuracy_mean = np.mean(accuracy_list)
accuracy_std = np.std(accuracy_list)
auc_roc_mean = np.mean(auc_roc_list)
auc_roc_std = np.std(auc_roc_list)
aupr_mean = np.mean(aupr_list)
aupr_std = np.std(aupr_list)


print(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
print(f"Recall: {recall_mean:.4f} ± {recall_std:.4f}")
print(f"F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")
print(f"Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
print(f"AUC-ROC: {auc_roc_mean:.4f} ± {auc_roc_std:.4f}")
print(f"AUPR: {aupr_mean:.4f} ± {aupr_std:.4f}")