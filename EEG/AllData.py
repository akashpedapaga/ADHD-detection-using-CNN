import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load data
adhd_data = pd.read_csv('ADHD1.csv', header=None)
control_data = pd.read_csv('Control1.csv', header=None)
f_adhd_data = pd.read_csv('FADHD.csv', header=None)
m_adhd_data = pd.read_csv('MADHD.csv', header=None)
f_control_data = pd.read_csv('FC.csv', header=None)
m_control_data = pd.read_csv('MC.csv', header=None)

# Add labels
adhd_data['Label'] = 1
control_data['Label'] = 0
f_adhd_data['Label'] = 1
m_adhd_data['Label'] = 1
f_control_data['Label'] = 0
m_control_data['Label'] = 0

# Find the maximum number of columns
max_columns = max(adhd_data.shape[1], control_data.shape[1], f_adhd_data.shape[1], m_adhd_data.shape[1], f_control_data.shape[1], m_control_data.shape[1])

def adjust_dimensions(df, max_columns):
    num_rows, num_cols = df.shape
    if num_cols < max_columns:
        # Pad with zeros if the number of columns is less than max_columns
        pad_width = max_columns - num_cols
        df = np.pad(df, ((0, 0), (0, pad_width)), 'constant', constant_values=0)
    elif num_cols > max_columns:
        # Trim if the number of columns is greater than max_columns
        df = df.iloc[:, :max_columns]
    return df

adhd_data = pd.DataFrame(adjust_dimensions(adhd_data.values, max_columns))
control_data = pd.DataFrame(adjust_dimensions(control_data.values, max_columns))
f_adhd_data = pd.DataFrame(adjust_dimensions(f_adhd_data.values, max_columns))
m_adhd_data = pd.DataFrame(adjust_dimensions(m_adhd_data.values, max_columns))
f_control_data = pd.DataFrame(adjust_dimensions(f_control_data.values, max_columns))
m_control_data = pd.DataFrame(adjust_dimensions(m_control_data.values, max_columns))

# Concatenate the datasets
data = pd.concat([adhd_data, control_data, f_adhd_data, m_adhd_data, f_control_data, m_control_data], ignore_index=True)

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
