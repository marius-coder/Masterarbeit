# -*- coding: cp1252 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam


data = pd.read_csv("Filtered_1hour.csv", sep=";", decimal=",", encoding= "cp1252",parse_dates=["Datetime"])
# Split the data into input/output sequences
input_seq = data[['Außentemperatur', 'Außenfeuchte',"Stunde","Werktag","Anwesenheit"]].values[:-1, :]
scaler = MinMaxScaler()
input_seq = scaler.fit_transform(input_seq)
output_seq = data[['Summe_Verbrauch']].values[1:]
scaler = MinMaxScaler()
output_seq = scaler.fit_transform(output_seq)
# Split the data into training and validation sets
train_input, test_input, train_output, test_output = train_test_split(input_seq, output_seq, test_size=0.2, shuffle=False)

# Reshape the input/output sequences for the LSTM
train_input = train_input.reshape((train_input.shape[0], 1, train_input.shape[1]))
test_input = test_input.reshape((test_input.shape[0], 1, test_input.shape[1]))


# Define the Keras model
def create_model(optimizer='adam', dropout_rate=0.0, neurons=100,learning_rate=0.0001,activation="relu"):


    model = Sequential()
    model.add(Dense(neurons, input_shape=(1,5), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
    return model

# Create a KerasRegressor wrapper for the scikit-learn API
model = KerasRegressor(build_fn=create_model, verbose=0)

# Define the hyperparameters to be tuned
param_grid = {
    'neurons': [50, 100, 200],
    #'dropout_rate': [0.0, 0.2, 0.5],
    #'optimizer': ['adam', 'sgd', 'rmsprop']    
    "epochs": [20,40,60],
    "learning_rate": [0.001,0.0005,0.0001],
    "activation": ["exponential","relu","leaky_relu"]
}

# Perform grid search cross-validation with 5-fold cross-validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# Fit the grid search to the training data
grid_result = grid.fit(train_input, train_output,batch_size=32)

# Print the best hyperparameters and corresponding mean test score
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, std, param))

# Predict using the best model
y_pred = grid_result.best_estimator_.predict(test_input)

# Compute the mean squared error on the test data
mse = mean_squared_error(test_output, y_pred)
print("Test MSE: %.3f" % mse)
