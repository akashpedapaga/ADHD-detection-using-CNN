'''
#This file is the fully functional running code with Reduced Datasets (Both IEEE and ADHD datasets with reduced size)
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
f_adhd_data = pd.read_csv('SamFADHD.csv', header=None)
m_adhd_data = pd.read_csv('SamMADHD.csv', header=None)
f_control_data = pd.read_csv('SamFC.csv', header=None)
m_control_data = pd.read_csv('SamMC.csv', header=None)

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

# LIME explainer
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.squeeze(X_train), 
    feature_names=list(range(X_train.shape[1])), 
    class_names=['Control', 'ADHD'], 
    verbose=True, 
    mode='classification'
)

# Choose a sample from the test set to explain
i = 0
exp = explainer.explain_instance(np.squeeze(X_test[i]), model.predict, num_features=10)
exp.show_in_notebook(show_table=True, show_all=False)

# Alternatively, you can save the explanation as an HTML file
exp.save_to_file('lime_explanation.html')

#Error in XAI Techniques
#Need to fix these errors

'''









import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from lime import lime_tabular

# Load data and add labels in a generic function
def load_data(filepath, label):
    df = pd.read_csv(filepath, header=None)
    df['Label'] = label
    return df

# Load datasets
adhd_data1 = load_data('ADHD1.csv', 1)
control_data1 = load_data('Control1.csv', 0)
adhd_data2 = load_data('ADHD2.csv', 1)
control_data2 = load_data('Control2.csv', 0)
f_adhd_data = load_data('SamFADHD.csv', 1)
m_adhd_data = load_data('SamMADHD.csv', 1)
f_control_data = load_data('SamFC.csv', 0)
m_control_data = load_data('SamMC.csv', 0)

# Combine all datasets
dataframes = [adhd_data1, control_data1, adhd_data2, control_data2, f_adhd_data, m_adhd_data, f_control_data, m_control_data]
data = pd.concat(dataframes, ignore_index=True)

# Shuffle the combined dataset
data = shuffle(data, random_state=42)

# Split the data
X = data.drop('Label', axis=1)
y = data['Label']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Data preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE to balance the training data
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Reshape data for 1D Convolutional Layer
X_train_res = np.expand_dims(X_train_res, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Model configuration
model = Sequential([
    Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(X_train_res.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train_res, y_train_res, epochs=20, batch_size=32, validation_split=0.2, shuffle=True, callbacks=[early_stopping])

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Test Accuracy: {accuracy}')

# Prediction and classification report
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# Setup LIME Explainer
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train,  # Make sure to use the 2D array version of X_train
    feature_names=['Feature %d' % i for i in range(X_train.shape[1])],
    class_names=['Control', 'ADHD'],
    mode='classification'
)

# Explain a prediction
i = np.random.randint(0, X_test.shape[0])  # Random test sample
explanation = explainer.explain_instance(X_test[i].reshape(-1), model.predict, num_features=10)
print('Explanation for test instance', i)
print(explanation.as_list())
