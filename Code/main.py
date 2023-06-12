import matplotlib
import matplotlip as matplotlip
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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

history = model.fit(x_train, y_train, epochs=500)

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
#fig.show()

fig2 = px.bar(df, x='Sleep (h)', y='Delta', color='Age',title='Time Perception Madness')
fig2.show()

df = pd.read_csv('TimePerceptionV6.csv', delimiter=',', encoding='utf-8')
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df['Sleep (h)'], y=df['Delta'], mode='markers', name ='Linear'))

coefficients = np.polyfit(df['Sleep (h)'], df['Delta'], 1)
line_of_best_fit = np.polyval(coefficients, df['Sleep (h)'])

fig3.add_trace(go.Scatter(x=df['Sleep (h)'], y=line_of_best_fit, line=dict(color='rgb(255,0,0)'), mode='lines', name='Line of Best Fit'))
fig3.update_layout(title='Time Perception Madness')
fig3.update_layout(xaxis_title='Sleep (h)', yaxis_title='Delta')
fig3.show()

#G = nx.Graph()

#Add nodes
#G.add_node('Berea College')
#G.add_nodes_from(range(1,10))

#Remove nodes
#G.remove_node('Berea College')
#G.remove_nodes_from([(1,2), (2,3)])

#adding and removing edges
#G.add_edge(5,100)
#G.remove_edge(5,100)
#G.add_edges_from()
#G.remove_edges_from()
#G.add_weighted_edges_from([(0,1,3.0), (1,4, 7.5)])
#G.add_path([1,2,3])

#nx.draw(G, with_labels=True)
#plt.draw()
#plt.show()

df = pd.read_csv("TimePerceptionV6.csv")
df.head()
TimePerception = nx.from_pandas_edgelist(df, source="Sex (1=male, 2=female)", target="Delta")
type(TimePerception)
#nx.info(us_graph)
TimePerception.nodes()
len(TimePerception.nodes())
TimePerception.edges()
len(TimePerception.edges())

#TimePerception.add_edge("Franklin", "Lincoln")
TimePerception.nodes()

nx.draw(TimePerception, with_labels=True, node_size=750, edge_color='aqua',
        node_color='Beige', font_size=8, font_color='black')
plt.draw()
plt.show()
