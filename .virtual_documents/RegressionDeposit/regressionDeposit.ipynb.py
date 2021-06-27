import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn
seaborn.set(style="ticks")


df = pd.read_excel("deposits.xlsx")


df.head()


df.describe()


def plot_grid(df=df, x="X", y="Y", hue="VMS Deposits"):
    """Plot facet grid of X,Y and associated deposits, count of volcanic rocks by type
    
    """
    dfcop = df.copy()
    dfcop.sort_values(by=[hue])
    fg = seaborn.FacetGrid(data=dfcop, hue=hue, palette="viridis", aspect=1.61)
    fg.map(plt.scatter, x, y).add_legend()
    fg.fig.suptitle(f'{hue} by Cell Coordinate')
    fg.set(ylim=(42, 55))
    fg.set(xlim=(22, 33))


plot_grid()


plot_grid(hue="V3")


sns.heatmap(df.corr(), annot=True)


msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]



print(len(train))
print(len(test))


from sklearn.linear_model import LinearRegression

def lsRegression(X="V1"):
    """Performs lsRegression to columns in DataFrame
    """
    # convert columns to np.array
    X = df[X].values.reshape(-1,1)
    Y = df["VMS Deposits"].values.reshape(-1,1)

    # Performing ordinary least square regression
    linear_regressor = LinearRegression() 
    linear_regressor.fit(X, Y)
    Y_pred = linear_regressor.predict(X)
    score, coeficient, intercept = linear_regressor.score(X, Y), linear_regressor.coef_, linear_regressor.intercept_
    return Y_pred, score, coeficient, intercept

def plotLSR(X, Y, Y_pred):
    """Plots Linear Square Regression of Y_pred
    """


Y_preds = {}
scores = []
coeficients = []
intercepts = []
for i in ["V1", "V2", "V3", "V4", "V5"]:
    Y_pred, score, coeficient, intercept = lsRegression(i)
    Y_preds[i] = Y_pred
    scores.append(score)
    coeficients.append(coeficient)
    intercepts.append(intercept)






fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharey=True)
Y = df["VMS Deposits"].values.reshape(-1,1)
for i, j in enumerate(["V1", "V2", "V3", "V4", "V5"]):
    rocks = df[j].values.reshape(-1,1)
    axs[i].scatter(rocks, Y)
    axs[i].plot(rocks, Y_preds[j], color="red")
    axs[i].set_ylabel("VMS Ocurrences")
    axs[i].set_xlabel(f"Number of Volcanic Rocks type {j}")
    
fig.tight_layout()
fig.suptitle('Linear Regression of 5 Volcanic Rock Types vs Presence of VMS Deposits')


scores


coeficients


intercepts


import statsmodels.api as sm


X = df[["V1", "V2", "V3", "V4", "V5"]] 
y = df["VMS Deposits"]
X = sm.add_constant(X) 

model = sm.OLS(y, X).fit() 
predictions = model.predict(X)


model.summary()


plt.plot(predictions)


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline




dataset = df.values
# split into input (X) and output (Y) variables
X = dataset[:,2:7]
Y = dataset[:,7]



# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=25, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: get_ipython().run_line_magic(".2f", " (%.2f) MSE\" % (results.mean(), results.std()))")


from sklearn.model_selection import train_test_split

X = dataset[:,2:7]
Y = dataset[:,7]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.1)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(5,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)


model.summary()



initial_weights = model.get_weights()

shuffle_weights(model, initial_weights)


from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

early_stop = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=10)
model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=128,epochs=100,callbacks=[early_stop])


losses = pd.DataFrame(model.history.history)
losses.plot()


from sklearn.metrics import mean_squared_error,mean_absolute_error

predictions = model.predict(X_test)
mean_absolute_error(y_test,predictions)



np.sqrt(mean_squared_error(y_test,predictions))
