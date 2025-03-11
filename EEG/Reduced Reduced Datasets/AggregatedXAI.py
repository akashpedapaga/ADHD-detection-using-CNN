'''
#FULL WORKING AGGREGATED XAI CODE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import lime
import lime.lime_tabular
from sklearn.utils.class_weight import compute_class_weight
import re

# Load data
adhd_data1 = pd.read_csv('F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedADHD1.csv', header=None)
control_data1 = pd.read_csv('F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedControl1.csv', header=None)
adhd_data2 = pd.read_csv('F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedADHD2.csv', header=None)
control_data2 = pd.read_csv('F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedControl2.csv', header=None)
f_adhd_data = pd.read_csv('F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedSamFADHD.csv', header=None)
m_adhd_data = pd.read_csv('F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedSamMADHD.csv', header=None)
f_control_data = pd.read_csv('F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedSamFC.csv', header=None)
m_control_data = pd.read_csv('F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedSamMC.csv', header=None)

# Add labels
adhd_data1['Label'] = 1
control_data1['Label'] = 0
adhd_data2['Label'] = 1
control_data2['Label'] = 0
f_adhd_data['Label'] = 1
m_adhd_data['Label'] = 1
f_control_data['Label'] = 0
m_control_data['Label'] = 0

# Ensure labels are integers
adhd_data1['Label'] = adhd_data1['Label'].astype(int)
control_data1['Label'] = control_data1['Label'].astype(int)
adhd_data2['Label'] = adhd_data2['Label'].astype(int)
control_data2['Label'] = control_data2['Label'].astype(int)
f_adhd_data['Label'] = f_adhd_data['Label'].astype(int)
m_adhd_data['Label'] = m_adhd_data['Label'].astype(int)
f_control_data['Label'] = f_control_data['Label'].astype(int)
m_control_data['Label'] = m_control_data['Label'].astype(int)

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
X = data.iloc[:, :max_columns - 1]  # Ensure we only take the original number of features
y = data.iloc[:, max_columns - 1].astype(int)  # Ensure labels are integers

# Check for negative labels and correct them
negative_label_indices = y < 0
if negative_label_indices.any():
    print(f"Negative labels found: {y[negative_label_indices]}")
    y[negative_label_indices] = 0  # Correct negative labels to 0

# Check class distribution
class_counts = y.value_counts()

# Perform train-test split without stratification if any class has fewer than 2 instances
if class_counts.min() < 2:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the training data if possible
apply_smote = all(count >= 2 for count in y_train.value_counts())
if apply_smote:
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
else:
    X_train_res, y_train_res = X_train, y_train

# Debugging: Print shapes and some samples of the data
print(f"X_train_res shape: {X_train_res.shape}")
print(f"y_train_res shape: {y_train_res.shape}")
print(f"Sample X_train_res: {X_train_res[:5]}")
print(f"Sample y_train_res: {y_train_res[:5]}")

# Check for negative values in y_train_res
unique_labels = np.unique(y_train_res)
print(f"Unique labels in y_train_res: {unique_labels}")

# Standardize the data
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Reshape data for 1D Convolutional Layer
X_train_res = np.expand_dims(X_train_res, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Debugging: Print shapes after reshaping
print(f"X_train_res shape after reshaping: {X_train_res.shape}")
print(f"X_test shape after reshaping: {X_test.shape}")

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_res), y=y_train_res)
class_weights = dict(enumerate(class_weights))

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

# Define the CNN model with modifications
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_res.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with focal loss
model.compile(optimizer=Adam(learning_rate=0.00001), loss=focal_loss(gamma=2., alpha=0.25), metrics=['accuracy'])

# Early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)

# Train the model with early stopping and learning rate reduction
history = model.fit(X_train_res, y_train_res, epochs=30, batch_size=32, validation_split=0.2, shuffle=True, callbacks=[early_stopping, reduce_lr], class_weight=class_weights)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Predict probabilities for the test set
y_pred_probs = model.predict(X_test)

# Adjust the decision threshold
threshold = 0.6  # Experiment with different thresholds
y_pred = (y_pred_probs > threshold).astype(int)

# Calculate and print classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Control', 'ADHD'], rotation=45)
plt.yticks(tick_marks, ['Control', 'ADHD'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_probs)
plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Plot training history
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.title('Training and Validation Loss/Accuracy')
plt.legend()
plt.show()

# LIME explanation for multiple instances
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_res[:, :, 0], feature_names=[f'Feature {i}' for i in range(X_train_res.shape[1])], class_names=['Control', 'ADHD'], discretize_continuous=True)

# Function to make predictions compatible with LIME
def predict_proba(X):
    probs = model.predict(np.expand_dims(X, axis=2))
    return np.hstack((1 - probs, probs))

# Explain multiple predictions and aggregate results
num_instances = 100  # Number of instances to explain
feature_importance = np.zeros(X_train_res.shape[1])

for i in range(num_instances):
    exp = explainer.explain_instance(X_test[i].reshape(-1), predict_proba, num_features=10)
    for feature, weight in exp.as_list():
        # Extract the feature index using regex to handle different formats
        match = re.search(r'Feature (\d+)', feature)
        if match:
            feature_index = int(match.group(1))
            if feature_index < 19:  # Ensure the index is within the original feature set range
                feature_importance[feature_index] += weight

# Average the feature importance
feature_importance /= num_instances

# Trim the feature importance to the original number of features
feature_importance = feature_importance[:19]

# Save the aggregated explanation to HTML file
html_file_path = 'lime_aggregated_explanation.html'
with open(html_file_path, 'w') as f:
    f.write('<html><body><h1>Aggregated LIME Explanation</h1><table border="1"><tr><th>Feature</th><th>Importance</th></tr>')
    for i, importance in enumerate(feature_importance):
        f.write(f'<tr><td>Feature {i}</td><td>{importance}</td></tr>')
    f.write('</table></body></html>')

# Print confirmation
print(f'Aggregated explanation saved to {html_file_path}')

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(19), feature_importance)  # Only plot the original number of features
plt.xlabel('Feature Index')
plt.ylabel('Average Importance')
plt.title('Aggregated Feature Importance')
plt.show()
'''



'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# Paths to datasets (replace with your actual file paths)
adhd_data1_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedADHD1.csv'
control_data1_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedControl1.csv'
adhd_data2_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedADHD2.csv'
control_data2_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedControl2.csv'
f_adhd_data_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedSamFADHD.csv'
m_adhd_data_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedSamMADHD.csv'
f_control_data_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedSamFC.csv'
m_control_data_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedSamMC.csv'

# Load datasets
adhd_data1 = pd.read_csv(adhd_data1_path, header=None)
control_data1 = pd.read_csv(control_data1_path, header=None)
adhd_data2 = pd.read_csv(adhd_data2_path, header=None)
control_data2 = pd.read_csv(control_data2_path, header=None)
f_adhd_data = pd.read_csv(f_adhd_data_path, header=None)
m_adhd_data = pd.read_csv(m_adhd_data_path, header=None)
f_control_data = pd.read_csv(f_control_data_path, header=None)
m_control_data = pd.read_csv(m_control_data_path, header=None)

# Add labels
adhd_data1['Label'] = 1
control_data1['Label'] = 0
adhd_data2['Label'] = 1
control_data2['Label'] = 0
f_adhd_data['Label'] = 1
m_adhd_data['Label'] = 1
f_control_data['Label'] = 0
m_control_data['Label'] = 0

# Ensure labels are integers
datasets = [adhd_data1, control_data1, adhd_data2, control_data2, f_adhd_data, m_adhd_data, f_control_data, m_control_data]
for dataset in datasets:
    dataset['Label'] = dataset['Label'].astype(int)

# Find the maximum number of columns
max_columns = max(dataset.shape[1] for dataset in datasets)

# Adjust dimensions of all datasets
def adjust_dimensions(df, max_columns):
    num_rows, num_cols = df.shape
    if num_cols < max_columns:
        pad_width = max_columns - num_cols
        df = np.pad(df, ((0, 0), (0, pad_width)), 'constant', constant_values=0)
    elif num_cols > max_columns:
        df = df.iloc[:, :max_columns]
    return pd.DataFrame(df)

datasets = [adjust_dimensions(dataset.values, max_columns) for dataset in datasets]

# Concatenate all datasets
data = pd.concat(datasets, ignore_index=True)

# Shuffle the data
data = shuffle(data)

# Extract features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1].astype(int)

# Check for missing values and handle them
if X.isnull().values.any() or np.isinf(X.values).any():
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

# Train-test split with stratification
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

# Build the CNN model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_res.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)

# Train the model
history = model.fit(X_train_res, y_train_res, epochs=30, batch_size=32, validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Generate ROC Curve
y_pred_probs = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})', color='orange')
plt.plot([0, 1], [0, 1], linestyle='--', color='blue')
plt.xlim([0.0, 0.05])  # Adjust zoom for clearer gap
plt.ylim([0.98, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Generate Confusion Matrix
y_pred = (y_pred_probs > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Control', 'ADHD'], rotation=45)
plt.yticks(tick_marks, ['Control', 'ADHD'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Print Classification Report
print(classification_report(y_test, y_pred))
'''




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Paths to datasets (replace with your actual file paths)
adhd_data1_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedADHD1.csv'
control_data1_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedControl1.csv'
adhd_data2_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedADHD2.csv'
control_data2_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedControl2.csv'
f_adhd_data_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedSamFADHD.csv'
m_adhd_data_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedSamMADHD.csv'
f_control_data_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedSamFC.csv'
m_control_data_path = 'F:\\Local Disk F\\MS CS\\Academics\\Thesis\\Code\\Adult-Adhd\\EEG\\Reduced Reduced Datasets\\RedSamMC.csv'

# Load datasets
adhd_data1 = pd.read_csv(adhd_data1_path, header=None)
control_data1 = pd.read_csv(control_data1_path, header=None)
adhd_data2 = pd.read_csv(adhd_data2_path, header=None)
control_data2 = pd.read_csv(control_data2_path, header=None)
f_adhd_data = pd.read_csv(f_adhd_data_path, header=None)
m_adhd_data = pd.read_csv(m_adhd_data_path, header=None)
f_control_data = pd.read_csv(f_control_data_path, header=None)
m_control_data = pd.read_csv(m_control_data_path, header=None)

# Add labels
adhd_data1['Label'] = 1
control_data1['Label'] = 0
adhd_data2['Label'] = 1
control_data2['Label'] = 0
f_adhd_data['Label'] = 1
m_adhd_data['Label'] = 1
f_control_data['Label'] = 0
m_control_data['Label'] = 0

# Ensure labels are integers
datasets = [adhd_data1, control_data1, adhd_data2, control_data2, f_adhd_data, m_adhd_data, f_control_data, m_control_data]
for dataset in datasets:
    dataset['Label'] = dataset['Label'].astype(int)

# Find the maximum number of columns
max_columns = max(dataset.shape[1] for dataset in datasets)

# Adjust dimensions of all datasets
def adjust_dimensions(df, max_columns):
    num_rows, num_cols = df.shape
    if num_cols < max_columns:
        pad_width = max_columns - num_cols
        df = np.pad(df, ((0, 0), (0, pad_width)), 'constant', constant_values=0)
    elif num_cols > max_columns:
        df = df.iloc[:, :max_columns]
    return pd.DataFrame(df)

datasets = [adjust_dimensions(dataset.values, max_columns) for dataset in datasets]

# Concatenate all datasets
data = pd.concat(datasets, ignore_index=True)

# Shuffle the data
data = shuffle(data)

# Extract features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1].astype(int)

# Check for missing values and handle them
if X.isnull().values.any() or np.isinf(X.values).any():
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

# Train-test split with stratification
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

# Build the CNN model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_res.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)

# Train the model
history = model.fit(X_train_res, y_train_res, epochs=30, batch_size=32, validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Generate ROC Curve
y_pred_probs = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})', color='orange')
plt.plot([0, 1], [0, 1], linestyle='--', color='blue')
plt.xlim([0.0, 0.05])  # Adjust zoom for clearer gap
plt.ylim([0.98, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Generate Precision-Recall Curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_probs)
plt.figure()
plt.plot(recall, precision, label='Precision-Recall Curve', color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Generate Confusion Matrix
y_pred = (y_pred_probs > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Control', 'ADHD'], rotation=45)
plt.yticks(tick_marks, ['Control', 'ADHD'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Print Classification Report
print(classification_report(y_test, y_pred))

# Plot Training and Validation Metrics
plt.figure()
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Training and Validation Metrics')
plt.legend()
plt.show()
