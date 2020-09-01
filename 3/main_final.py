import numpy as np
import csv 
from UtilityFunctions import ReadData_char
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


train_features, train_labels, test_features = ReadData_char()

ordinalEncoder = OrdinalEncoder(dtype= np.int64)
ordinalEncoder.fit(train_features)
train_enc = ordinalEncoder.transform(train_features)
test_enc = ordinalEncoder.transform(test_features)

one = OneHotEncoder()
one.fit(train_enc)
train_enc = one.transform(train_enc).toarray()
test_enc = one.transform(test_enc).toarray()


def neuralNet():
    
    std_scaler = StandardScaler()
    
    std_scaler.fit(train_enc)

    X_train_trsf = std_scaler.transform(train_enc)
    X_test_trsf = std_scaler.transform(test_enc)

    mlpclass = MLPClassifier(hidden_layer_sizes=(1000), verbose=10, max_iter=1000, tol=0.000001, alpha=0.000001, batch_size='auto', random_state=10)

    mlpclass.fit(X_train_trsf, train_labels)

    y_pred = mlpclass.predict(X_test_trsf)
    
    return y_pred
        
y_pred = neuralNet()

output_data = []

for i in range (len(y_pred)):
    output_data.append([y_pred[i]])
    
#we return the data to the csv file   
filename = "result_final_SGS.csv"
  
# writing to csv file 
with open(filename, 'w') as csvfile:
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile)    
    # writing the data rows 
    csvwriter.writerows(output_data)
