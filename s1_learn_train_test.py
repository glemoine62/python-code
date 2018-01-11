from __future__ import print_function

import numpy as np
import tflearn
import sys

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv


data, labels = load_csv(sys.argv[1], target_column=-1,
                        categorical_labels=True, n_classes=3)

test_data, test_labels = load_csv(sys.argv[2], target_column=-1,
                        categorical_labels=True, n_classes=3)

org_data, org_labels = load_csv(sys.argv[2], target_column=-1,
                        categorical_labels=True, n_classes=3)

print(len(test_data))
# Preprocessing function
def preprocess(profiles, columns_to_delete):
    # Sort by descending id and delete columns
    for column_to_delete in sorted(columns_to_delete, reverse=True):
        [profile.pop(column_to_delete) for profile in profiles]
    return np.array(profiles, dtype=np.float32)

# Ignore 'id' column
to_ignore=[0]

# Preprocess data
data = preprocess(data, to_ignore)
test_data = preprocess(test_data, to_ignore)
# Build neural network
net = tflearn.input_data(shape=[None, len(data[0])])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=80, batch_size=32, show_metric=True)

# Check predictions for the samples not used in training
for i in range(0,len(test_data)):
  sample = test_data[i]
  #print(labels[i])
  # Predict surviving chances (class 1 results)
  pred = model.predict([sample])
  print("{:s},{:d},{:6.2f},{:6.2f},{:6.2f}".format(org_data[i][0], test_labels[i].tolist().index(1.0), 
    100*pred[0][0], 100*pred[0][1], 100*pred[0][2]))
