import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from keras.constraints import maxnorm

def create_model(learn_rate, 
                 init_mode,
                 activation,
                 neurons):
    model = Sequential()
    # 1st layer 
    model.add(Dense(neurons, input_dim=2, kernel_initializer=init_mode, 
                    activation=activation))
    model.add(Dense(1, kernel_initializer=init_mode, activation=activation))

    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer="Adam", metrics=[keras.metrics.BinaryAccuracy()])
    
    return model

model = KerasClassifier(model=create_model, verbose=True,
                        optimizer="Adam", learn_rate=0.001, 
                        loss=keras.losses.BinaryCrossentropy(), epochs=10,
                        batch_size=10, activation="sigmoid", 
                        init_mode="uniform", neurons=1)

# GridSearchCV to test for several hyperparameters
hyper_params = {
    'batch_size': ([10, 20, 50]),
    'epochs': ([10, 20, 50]),
    'learn_rate': ([0.001, 0.01, 0.1]),
    'activation': (['softmax', 'tanh', 'sigmoid']),
    'neurons': ([10, 15, 20, 25, 30]) 
}

grid = GridSearchCV(estimator=model, param_grid=hyper_params, n_jobs=-1, cv=3, verbose=3, error_score="raise")
grid_result = grid.fit(inputs, targets)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#Best: 0.676359 using {'activation': 'softsign', 'batch_size': 10, 'epochs': 50, 'learn_rate': 0.1, 'neurons': 15}
def create_model(learn_rate, 
                 init_mode,
                 activation,
                 neurons):
    model = Sequential()
    # 1st layer 
    model.add(Dense(neurons, input_dim=2, kernel_initializer=init_mode, 
                    activation=activation))
    model.add(Dense(neurons - 4, activation=activation))
    model.add(Dense(1, kernel_initializer=init_mode, activation=activation))

    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer="Adam", metrics=["accuracy"])
    
    return model

model = KerasClassifier(model=create_model, verbose=True,
                        optimizer="Adam", learn_rate=0.1, 
                        loss=keras.losses.BinaryCrossentropy(), epochs=50,
                        batch_size=10, activation="softsign", 
                        init_mode="uniform", neurons=1)

# GridSearchCV to test for several hyperparameters
hyper_params = {
    'batch_size': ([10, 20, 40, 60]),
    'epochs': ([10, 20, 50]),
    'learn_rate': ([0.001, 0.01, 0.1]),
    'neurons': ([10, 15, 20, 25, 30]) 
}

grid = GridSearchCV(estimator=model, param_grid=hyper_params, n_jobs=-1, cv=3, verbose=3, error_score="raise")
grid_result = grid.fit(inputs, targets)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))