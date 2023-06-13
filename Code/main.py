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
from collections import Counter
import matplotlib.cm as cm


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

history = model.fit(x_train, y_train, epochs=100)

model.evaluate(x_test, y_test)

history.history.keys()

z = 'accuracy'
print(z)




df = pd.read_csv('TimePerceptionV6.csv')

fig = px.bar(df, x='Sleep (h)', y='Delta', title='Time Perception Madness')
#fig.show()

fig2 = px.bar(df, x='Sleep (h)', y='Delta', color='Sex',title='Time Perception Madness')
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

# ------------- Node Diagrams ------------
df = pd.read_csv("TimePerceptionV6.csv")
df.head()
TimePerception = nx.from_pandas_edgelist(df, source="Sleep (h)", target="Age")
type(TimePerception)

# Calculate node frequencies based on occurrence in the DataFrame
node_frequencies = pd.concat([df["Sleep (h)"], df["Age"]]).value_counts().to_dict()
node_frequencies2 = pd.concat([df["Sleep (h)"], df["Age"]]).value_counts().to_dict()

TimePerception.nodes()
len(TimePerception.nodes())
TimePerception.edges()
len(TimePerception.edges())
TimePerception.nodes()


# Calculate node sizes based on frequencies
node_sizes = [node_frequencies.get(node, 1) * 250 for node in TimePerception.nodes()]
edge_width = [node_frequencies.get(node, 1) * 0.25 for node in TimePerception.nodes()]

nx.draw(TimePerception, width = edge_width, with_labels=True, node_size=node_sizes, edge_color = 'red', edgecolors='grey',
        node_color='beige', font_size=8, font_color='black')
plt.draw()
plt.show()
