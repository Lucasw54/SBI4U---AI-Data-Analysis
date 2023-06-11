import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import plotly.express as px


dataset = pd.read_csv('TimePerceptionV6.csv')

x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'))
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(tf.keras.layers.Dense(12, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=1000)

model.evaluate(x_test, y_test)

history.history.keys()

z = 'accuracy'
print(z)

#plt.plot(history.history['accuracy'])
#plt.plot(history.history['loss'])
#plt.title('model accuracy/loss')
#plt.ylabel('loss | accuracy')
#plt.xlabel('epoch')
#plt.legend(['accuracy', 'loss'], loc='upper left')
#plt.show()

df = pd.read_csv('TimePerceptionV6.csv')

fig = px.bar(df, x='Sleep (h)', y='Delta', title='Time Perception Madness')
fig.show()
