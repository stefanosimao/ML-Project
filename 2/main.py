import numpy as np
from sklearn.svm import SVC
import csv 


def main():
    
    #TRAIN & TEST DATA
    data_train = np.genfromtxt('train.csv', delimiter = ',')
    data_test = np.genfromtxt('test.csv', delimiter = ',')
    
    #get index for the test 
    id_test= data_test[1:,0]
    #remove index col
    data_train = data_train[1:,1:]
    data_test = data_test[1:,1:]
    #isolating x and y
    x_train = data_train[:,1:]
    y_train = data_train[:,0]
    x_test = data_test[:,0:]
    
    
    #we use the Support Vector Classification function from the sklearn.svm library
    classification = SVC(kernel='rbf', gamma='scale', decision_function_shape= 'ovr')
    #fit the train data to the classification
    classification.fit(x_train,y_train)
    #predict the y from the test data x
    y_test = classification.predict(x_test)
    
    #here we put together the y that we got from the prediction with the id and the field    
    output_data = []
    
    for i in range (len(y_test)):
        output_data.append([id_test[i],y_test[i]])
    
    fields = ['Id', 'y'] 
    
    
    #we return the data to the csv file
    
    filename = "test_result_final.csv"
  
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
      
        # writing the fields 
        csvwriter.writerow(fields) 
      
        # writing the data rows 
        csvwriter.writerows(output_data)
    
    
    return

if __name__ == "__main__":

    main()