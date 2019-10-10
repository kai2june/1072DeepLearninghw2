import scipy.io
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# load file
fisheriris = scipy.io.loadmat('fisheriris.mat')

############# iris
# split into 80%training, 20%testing
iris_train_feature, iris_test_feature, iris_train_label, iris_test_label = \
    train_test_split(fisheriris['meas'], fisheriris['species'],  test_size=0.2, shuffle=True, random_state=113)

for i in range(len(iris_train_label)):
    iris_train_label[i] = iris_train_label[i][0]
for i in range(len(iris_test_label)):
    iris_test_label[i] = iris_test_label[i][0]
iris_train_label = iris_train_label.ravel()
iris_test_label = iris_test_label.ravel()

# encode class values as integer
encoder = LabelEncoder()
encoder.fit(iris_train_label)
encoded_iris_train_label = encoder.transform(iris_train_label)

encoder.fit(iris_test_label)
encoded_iris_test_label = encoder.transform(iris_test_label)

# convert label from integer to one-hot code
one_hot_iris_train_label = np_utils.to_categorical(encoded_iris_train_label)
one_hot_iris_test_label = np_utils.to_categorical(encoded_iris_test_label)

# neural net
model = Sequential()
model.add(Dense(input_dim=4, units=128,  activation='relu'))
for i in range(4):
    model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=iris_train_feature, y=one_hot_iris_train_label, batch_size=5, epochs=10)
result = model.evaluate(iris_train_feature, one_hot_iris_train_label, batch_size=30)
print("iris_train_NN accuracy=", result[1])

result = model.evaluate(iris_test_feature, one_hot_iris_test_label, batch_size=30)
print("iris_test_NN accuracy=", result[1])