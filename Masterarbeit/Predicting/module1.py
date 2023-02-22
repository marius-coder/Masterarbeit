# -*- coding: cp1252 -*-



import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, CuDNNLSTM, Bidirectional,SimpleRNN,BatchNormalization, Dropout
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

data = pd.read_csv("Filtered_15min.csv", sep=";", decimal=",", encoding= "cp1252",parse_dates=["Datetime"])

# Split the data into input/output sequences
input_seq = data[['Außentemperatur', 'Außenfeuchte',"Stunde","Werktag","Anwesenheit"]].values[:-1, :]
input_seq = (input_seq-input_seq.mean())/input_seq.std()
#scaler = MinMaxScaler()
#input_seq = scaler.fit_transform(input_seq)
output_seq = data[['Summe_Verbrauch']].values[1:]
output_seq =(output_seq-output_seq.mean())/output_seq.std()
#scaler = MinMaxScaler()
#output_seq = scaler.fit_transform(output_seq)
# Split the data into training and validation sets
train_input, test_input, train_output, test_output = train_test_split(input_seq, output_seq, test_size=0.2, shuffle=False)

# Reshape the input/output sequences for the LSTM
train_input = train_input.reshape((train_input.shape[0], 1, train_input.shape[1]))
test_input = test_input.reshape((test_input.shape[0], 1, test_input.shape[1]))



def accuracy(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

# Build the LSTM model
layer1 = Input(shape= (1,5))
layer2 = LSTM(32, return_sequences=True) (layer1)
layer3 = LSTM(16) (layer2)
layer4 = Dense(8) (layer3)
output = Dense(1, activation= "linear") (layer4)

model = Model(inputs= layer1, outputs= output)
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0005), metrics=[accuracy])

# Train the model on the input/output sequences
history = model.fit(train_input, train_output, epochs=60, batch_size=8, verbose=2, validation_split=0.1)

model.save("my_model.h5")

score, acc = model.evaluate(test_input, test_output)
print('Test score:', score)
print('Test accuracy:', acc)


#Plot Stuff
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.plot(history.history['accuracy'], color='black', label='training accuracy')
ax1.plot(history.history['val_accuracy'], color='grey', label='validation accuracy')
ax1.legend()

ax2 = ax1.twinx()
ax2.set_ylabel('Loss')
ax2.plot(history.history['loss'], color='red', label='training loss')
ax2.plot(history.history['val_loss'], color='orange', label='validation loss')
ax2.legend(loc='center right')

plt.show()



predicted = model.predict(test_input)
source = ColumnDataSource(data=dict(
    time=data["Datetime"][len(data)-len(test_output):],#np.linspace(0,len(test_output),len(test_output)+1),
    actual=test_output[:, 0],
    predicted=predicted[:, 0]
))

# create the figure and plot the actual and predicted values
p = figure(width=1600, height=800)
#p.line(x='time', y=, line_width=2, source=source, legend_label='Actual')
p.scatter(x='actual', y='predicted', line_width=2, source=source, color='red', legend_label='Predicted')

# customize the plot
p.title.text = 'Actual vs. Predicted Values'
p.legend.location = 'top_left'
p.legend.click_policy = 'hide'

# show the plot
show(p)

# create the figure and plot the actual and predicted values
p2 = figure(width=1600, height=800)
p2.line(x='time', y="Actual", line_width=2, source=source, legend_label='actual')
p2.line(x='time', y="predicted", line_width=2, source=source, legend_label='predicted')

# customize the plot
p2.title.text = 'Actual vs. Predicted Values'
p2.legend.location = 'top_left'
p2.legend.click_policy = 'hide'

# show the plot
show(p2)



















