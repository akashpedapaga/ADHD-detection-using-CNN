import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import shap
from sklearn.utils.class_weight import compute_class_weight

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
X_train_res = np.expand_dims(X_train_res, axis=2)  # Shape should be (samples, features, 1)
X_test = np.expand_dims(X_test, axis=2)  # Shape should be (samples, features, 1)

# Check shapes
print(f"X_train_res shape: {X_train_res.shape}")
print(f"X_test shape: {X_test.shape}")

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

# Define the CNN model with modifications to reduce overfitting
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=(X_train_res.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with focal loss and a lower learning rate for better generalization
model.compile(optimizer=Adam(learning_rate=0.0001), loss=focal_loss(gamma=2., alpha=0.25), metrics=['accuracy', 'Precision'])

# Early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model with early stopping and learning rate reduction
history = model.fit(X_train_res, y_train_res, epochs=50, batch_size=64, validation_split=0.2, shuffle=True, callbacks=[early_stopping, reduce_lr], class_weight=class_weights)

# Evaluate the model
test_loss, test_accuracy, test_precision = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')
print(f'Test Precision: {test_precision}')

# Predict probabilities for the test set
y_pred_probs = model.predict(X_test)

# Adjust the decision threshold to handle class imbalance
y_pred = (y_pred_probs > 0.7).astype(int)  # Increased threshold to reduce false positives

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

# SHAP explanation
# Flatten the test data for SHAP
X_test_flat = X_test.squeeze(axis=2)

# Sample a subset of the test data for SHAP calculations
sample_size = 100  # Adjust this to a smaller number to reduce computation
X_test_sample = X_test_flat[:sample_size]

# Initialize SHAP explainer with a subset of training data to improve performance
explainer = shap.KernelExplainer(lambda x: model.predict(np.expand_dims(x, axis=2)), X_train_res[:100].squeeze(axis=2))

# Calculate SHAP values for the sample of the test set
shap_values = explainer.shap_values(X_test_sample)

# Ensure SHAP values have the same length as the number of features
shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values

# Plot summary plot using the sampled data
shap.summary_plot(shap_values, X_test_sample)

# SHAP feature importance
shap_importance = np.abs(shap_values).mean(axis=0)  # Ensure correct dimensionality
shap_importance_df = pd.DataFrame({
    'Feature': [f'Feature {i}' for i in range(X_test_sample.shape[1])],
    'Importance': shap_importance
})

# Sort the feature importance by descending order
shap_importance_df = shap_importance_df.sort_values(by='Importance', ascending=False)

# Display SHAP importance as a table
print(shap_importance_df)

# SHAP summary plot
shap.summary_plot(shap_values, X_test_sample, plot_type="bar")
