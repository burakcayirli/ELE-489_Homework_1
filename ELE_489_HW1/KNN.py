import numpy as np
from collections import Counter

def distance(point1, point2):
  return np.sqrt(np.sum((point1 - point2)**2))  #euclidian distances 
  #return np.sum(np.abs(point1 - point2)) #manhattan distance

def KNN(X_train, X_test, Y_train, Y_test, K): #it takes the datas
    
    predicted = []
    X_train_np = X_train.to_numpy()
    Y_train_np = Y_train.to_numpy()

    for test_point in X_test.to_numpy():
        
        distances = [distance(train_point, test_point) for train_point in X_train_np] #calculates all distances from test points
        distance_indices = np.argsort(distances)[:K]                                  #sorts the distances and takes first Kth distance
        K_nearest_labels = [Y_train_np[i] for i in distance_indices]                  #it determines the labels
        most_common = Counter(K_nearest_labels).most_common(1)[0][0]                  #find the most common label
        predicted.append(int(most_common))                                            #append it into predicted class

    return predicted                                                                  #returns the predicted class