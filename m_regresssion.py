%importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

%loading data
data = pd.read_csv(‘data.csv’) %pandas function to read csv file
data.head() %shows the first five rows of the data
data.info() %shows information about the data
sns.pairplot(advert, x_vars =[‘A’,’B’,’C’], y_vars=[‘D’],height=7,aspect=0.7) %single function giving sub plots showing relationship between individual predictor and target

X= advert[[‘A’,’B’,’C’]]
Y= advert.D

X_train,  X_test, Y_train, Y_test = train_test_split(X,Y, random_state =1) %splitting dataset in training and testing data
lm1  = LinearRegression().fit(X_train,Y_train)
Print(lm1.intercept_) %gives the value of intercept of the model
Print(lm1.coef_)  %gives values of the coefficients of the model

List(zip([[‘A’,’B’,’C’]],lm1_coef_)) % gives coefficients corresponding to the feature

sns.heatmap(advert.corr(),annot = True) %shows correlation among feature variables also the output 

lm1_preds = lm1.predict(X_test) %prediction

Print(“RMSE:”, np.sqrt(mean_squared_error(y_test, lm1_preds))) %root mean squared error calculation (minimum)
Print(“R^2: ”, r2_score(y_test, lm1_preds)) %R square (maximum)

from yellowbrick.regressor import PredictionError, Residualsplot

Visualizer = PredictionError(lm5).fit(X_train, Y_train)
Visualizer.score(x_test,y_test)
Visualizer.poof; %visualizing the output
