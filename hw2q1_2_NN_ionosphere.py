import scipy.io
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# load file
ionosphere = scipy.io.loadmat('ionosphere.mat')

############# ionosphere
# split into 80%training, 20%testing
iono_train_feature, iono_test_feature, iono_train_label, iono_test_label = \
    train_test_split(ionosphere['X'], ionosphere['Y'], test_size=0.2, shuffle=True, random_state=113)

for i in range(len(iono_train_label)):
    iono_train_label[i] = iono_train_label[i][0]
for i in range(len(iono_test_label)):
    iono_test_label[i] = iono_test_label[i][0]
iono_train_label = iono_train_label.ravel()
iono_test_label = iono_test_label.ravel()

# encode class values as integer
encoder = LabelEncoder()
encoder.fit(iono_train_label)
encoded_iono_train_label = encoder.transform(iono_train_label)

encoder.fit(iono_test_label)
encoded_iono_test_label = encoder.transform(iono_test_label)

# convert label from integer to one-hot code
one_hot_iono_train_label = np_utils.to_categorical(encoded_iono_train_label)
one_hot_iono_test_label = np_utils.to_categorical(encoded_iono_test_label)

# neural net
model = Sequential()
model.add(Dense(input_dim=34, units=128,  activation='relu'))
for i in range(4):
    model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=iono_train_feature, y=one_hot_iono_train_label, batch_size=5, epochs=10)
result = model.evaluate(iono_train_feature, one_hot_iono_train_label, batch_size=30)
print("iono_train_NN accuracy=", result[1])

result = model.evaluate(iono_test_feature, one_hot_iono_test_label, batch_size=30)
print("iono_test_NN accuracy=", result[1])