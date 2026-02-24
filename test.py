from ucimlrepo import fetch_ucirepo 
import numpy as np

## Read in the data (however you prefer), and modify the target from a categorical variable to a binary target [0,1] #
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 

## Store the input features (x_input) in one array and the target features (t_target) in a separate array #
X = predict_students_dropout_and_academic_success.data.features 
Y = predict_students_dropout_and_academic_success.data.targets 

x_input = X.to_numpy()
t_target = Y.to_numpy()
t_target[t_target == 'Dropout'] = 0
t_target[t_target == 'Graduate'] = 1
t_target[t_target == 'Enrolled'] = 1
np.set_printoptions(threshold=np.inf, linewidth=200)   
print(t_target)
