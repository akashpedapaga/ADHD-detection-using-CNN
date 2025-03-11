import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# Load data
adhd_data1 = pd.read_csv('F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\Reduced Reduced Datasets\RedADHD1.csv', header=None)
control_data1 = pd.read_csv('F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\Reduced Reduced Datasets\RedControl1.csv', header=None)
adhd_data2 = pd.read_csv('F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\Reduced Reduced Datasets\RedADHD2.csv', header=None)
control_data2 = pd.read_csv('F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\Reduced Reduced Datasets\RedControl2.csv', header=None)
f_adhd_data = pd.read_csv('F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\Reduced Reduced Datasets\RedSamFADHD.csv', header=None)
m_adhd_data = pd.read_csv('F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\Reduced Reduced Datasets\RedSamMADHD.csv', header=None)
f_control_data = pd.read_csv('F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\Reduced Reduced Datasets\RedSamFC.csv', header=None)
m_control_data = pd.read_csv('F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\Reduced Reduced Datasets\RedSamMC.csv', header=None)

# Add labels
adhd_data1['Label'] = 1
control_data1['Label'] = 0
adhd_data2['Label'] = 1
control_data2['Label'] = 0
f_adhd_data['Label'] = 1
m_adhd_data['Label'] = 1
f_control_data['Label'] = 0
m_control_data['Label'] = 0

# Find the maximum number of columns
max_columns = max(
    adhd_data1.shape[1], control_data1.shape[1], adhd_data2.shape[1], control_data2.shape[1], 
    f_adhd_data.shape[1], m_adhd_data.shape[1], f_control_data.shape[1], m_control_data.shape[1]
)

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

adhd_data1 = pd.DataFrame(adjust_dimensions(adhd_data1.values, max_columns))
control_data1 = pd.DataFrame(adjust_dimensions(control_data1.values, max_columns))
adhd_data2 = pd.DataFrame(adjust_dimensions(adhd_data2.values, max_columns))
control_data2 = pd.DataFrame(adjust_dimensions(control_data2.values, max_columns))
f_adhd_data = pd.DataFrame(adjust_dimensions(f_adhd_data.values, max_columns))
m_adhd_data = pd.DataFrame(adjust_dimensions(m_adhd_data.values, max_columns))
f_control_data = pd.DataFrame(adjust_dimensions(f_control_data.values, max_columns))
m_control_data = pd.DataFrame(adjust_dimensions(m_control_data.values, max_columns))

# Concatenate the datasets
data = pd.concat([adhd_data1, control_data1, adhd_data2, control_data2, f_adhd_data, m_adhd_data, f_control_data, m_control_data], ignore_index=True)

# Shuffle the data
data = shuffle(data)

# Extract features and labels
X = data.iloc[:, :data.shape[1] - 1]
y = data.iloc[:, -1]

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the training data
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Standardize the data
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Reshape data for 1D Convolutional Layer
X_train_res = np.expand_dims(X_train_res, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Focal Loss function
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = K.cast(y_true, K.floatx())
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(fl)
    return focal_loss_fixed

# Define the CNN model
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(X_train_res.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with focal loss
model.compile(optimizer=Adam(learning_rate=0.00001), loss=focal_loss(gamma=2., alpha=0.25), metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train_res, y_train_res, epochs=20, batch_size=32, validation_split=0.2, shuffle=True, callbacks=[early_stopping])

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Test Accuracy: {accuracy}')

# Predict probabilities for the test set
y_pred_probs = model.predict(X_test)

# Convert probabilities to class labels
y_pred = (y_pred_probs > 0.5).astype(int)

# Calculate and print classification report
print(classification_report(y_test, y_pred))

