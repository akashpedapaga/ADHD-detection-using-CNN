import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import lime
from lime import lime_tabular

# Load data
adhd_data = pd.read_csv('ADHD1.csv', header=None)
control_data = pd.read_csv('Control1.csv', header=None)

# Assuming the label column is 'Label', where 1 represents ADHD and 0 represents control
adhd_data['Label'] = 1
control_data['Label'] = 0

# Concatenate the datasets
data = pd.concat([adhd_data, control_data], ignore_index=True)

# Shuffle the data
data = shuffle(data)

# Extract features and labels
X = data.iloc[:, :data.shape[1] - 1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for 1D Convolutional Layer
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Define the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, shuffle=True)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Test Accuracy: {accuracy}')

# Predict probabilities for the test set
y_pred_probs = model.predict(X_test)

# Convert probabilities to class labels
y_pred = (y_pred_probs > 0.5).astype(int)

# Calculate and print classification report
print(classification_report(y_test, y_pred))

# Using LIME for explanations
# Flatten the data back for LIME since LIME works with tabular data
X_train_flat = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test_flat = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Define a function to make predictions for LIME
def predict_fn(data):
    data = data.reshape(data.shape[0], data.shape[1], 1)
    preds = model.predict(data)
    return np.hstack((1 - preds, preds))

# Create a LIME explainer
explainer = lime_tabular.LimeTabularExplainer(X_train_flat, feature_names=[f'feature_{i}' for i in range(X_train_flat.shape[1])], class_names=['Control', 'ADHD'], verbose=True, mode='classification')

# Explain a specific instance from the test set
i = 0  # Choose the instance index you want to explain
exp = explainer.explain_instance(X_test_flat[i], predict_fn, num_features=10)

# Display the explanation
exp.show_in_notebook(show_table=True, show_all=False)
