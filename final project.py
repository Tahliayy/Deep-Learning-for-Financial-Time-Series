import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.cluster import KMeans
from minisom import MiniSom
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from pylab import bone, pcolor, colorbar, plot, show
from tensorflow.keras.metrics import Precision
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, balanced_accuracy_score


# Set the environment variable to control MKL behavior
os.environ['OMP_NUM_THREADS'] = '5'

# Load the data
data = pd.read_csv('C:/Users/Lenovo/Desktop/CQF/final/merged_final_data_with_financials.csv')

# Convert the 'Date' column to datetime format and exclude it from ML operations
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
X = data.drop(['Date'], axis=1)  # Exclude 'Date' from features

# Calculate daily log returns
X['Log_Returns'] = np.log(X['Price'] / X['Price'].shift(1))
# Fill the NaN value in the first row with the value from the second row for Log_Returns
X['Log_Returns'].fillna(method='bfill', inplace=True)

# Calculate EWMA of log returns
lambda_ = 0.94
X['EWMA_Volatility'] = X['Log_Returns'].ewm(alpha=1 - lambda_).std() * np.sqrt(252)

# Fill the NaN value in the first row for EWMA_Volatility with the value from the second row
X['EWMA_Volatility'].fillna(method='bfill', inplace=True)

# Convert all other columns to appropriate numeric types, ignoring errors
X = X.apply(pd.to_numeric, errors='coerce')

# Check the number of missing values in each column of the dataset
missing_values = X.isnull().sum()
# Print the number of missing values per column
print("Missing values in each column:\n", missing_values)

missing_values = missing_values[missing_values > 0]  # Filter to show only columns with missing values

# Print columns with missing values and their respective counts
if missing_values.empty:
    print("No missing values found in any column.")
else:
    print("Missing values in columns:\n", missing_values)


# Check data types and descriptive statistics
print(X.dtypes)
print(X.describe())

# Plot histograms for selected features
features_to_plot = ['Price', 'CVol', 'Change', '% Change', '% Return',
                    'Log_Returns', 'Total Return (Gross)', 'Amount', 'Open', 'High', 'Low', 'EWMA_Volatility']
X[features_to_plot].hist(figsize=(20, 15), bins=50, layout=(4, 3))
plt.tight_layout()
plt.show()

# Calculate the correlation matrix for all features
correlation_matrix = X.corr()
plt.figure(figsize=(60, 60))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix for All Features')
plt.show()

# Identify and remove highly correlated features
def correlated_features(data, threshold=0.9):
    col_corr = set()
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

highly_correlated_features = correlated_features(X)
X = X.drop(columns=highly_correlated_features)
print("Removed Features:", highly_correlated_features)
print("Remaining Features:", X.columns)

# Calculate the correlation matrix for the remaining features
remaining_corr_matrix = X.corr()

# Plot the heatmap for the correlation matrix
plt.figure(figsize=(10, 8))  # Adjust the size according to your needs
sns.heatmap(remaining_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Remaining Features')
plt.show()


# Add 'Return sign' based on '% Change'
threshold = 0.25
X['Return sign'] = np.where(X['% Change'] > threshold, 1, 0)
Y = X['Return sign']

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(X)

# K-means clustering
kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
kmeans.fit(features_scaled)
X['Kmeans_Cluster'] = kmeans.labels_

# Initialize and train SOM
som = MiniSom(x=5, y=5, input_len=features_scaled.shape[1], sigma=0.5, learning_rate=0.5)
som.random_weights_init(features_scaled)
som.train_random(features_scaled, num_iteration=100)

# Visualize SOM
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's', 'D', '^', 'p']
colors = ['r', 'g', 'b', 'c', 'm']

for idx, x in enumerate(features_scaled):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[X['Kmeans_Cluster'][idx]],
         markeredgecolor=colors[X['Kmeans_Cluster'][idx]], markerfacecolor='None',
         markersize=10, markeredgewidth=2)
show()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, Y)
# Get feature importances and compute a threshold
feature_importances = model.feature_importances_
threshold = sorted(feature_importances, reverse=True)[19]  # Get the threshold for the 20th most important feature

# Use SelectFromModel to select features
selector = SelectFromModel(model, threshold=threshold)
selector.fit(X, Y)
selected_features = X.columns[selector.get_support()]

# Retain only the selected features
X_reduced = X[selected_features]

print("Selected features:", selected_features)

# Ensure all data is float
X_reduced = X_reduced.select_dtypes(include=[np.number])

# Function to create dataset with lookback
def create_dataset(X, Y, time_steps=30, delay=2):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - delay):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(Y[i + time_steps + delay])
    return np.array(Xs), np.array(ys)

# Define time_steps and delay
time_steps = 30
delay = 2

# Prepare input and output columns
feature_columns = selected_features  # Assuming 'selected_features' is defined and contains relevant features
target_column = 'Return sign'  # Ensure this is defined and part of 'data'

# Convert feature_columns to indices
feature_indices = [X_reduced.columns.get_loc(col) for col in feature_columns]
target_index = X_reduced.columns.get_loc(target_column)
X, Y = create_dataset(X_reduced.values[:, feature_indices], X_reduced.values[:, target_index], time_steps, delay)
# Splitting the data into train, validation, and test sets (60%, 20%, 20%)
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(time_steps, len(feature_columns))))
model.add(LSTM(100, activation='relu'))  # Additional LSTM layer for depth
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=[Precision(name='precision')])


# Set the learning rate to decrease automatically
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

# Train the model
history = model.fit(X_train, Y_train, epochs=80, batch_size=8, validation_data=(X_val, Y_val), verbose=1, callbacks=[reduce_lr])

# Generate predictions for evaluation
Y_val_pred = model.predict(X_val)
Y_test_pred = model.predict(X_test)
Y_val_pred_bin = (Y_val_pred > 0.5).astype(int)
Y_test_pred_bin = (Y_test_pred > 0.5).astype(int)


# Evaluate the model on the validation set (newly added)
val_performance = model.evaluate(X_val, Y_val, verbose=0)
print(f'Validation Precision: {val_performance[1]}')

# Evaluate the model on the test set
test_performance = model.evaluate(X_test, Y_test, verbose=0)
print(f'Test Precision: {test_performance[1]}')


# Plotting model loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot training and validation Precision
plt.figure(figsize=(10, 5))
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.title('Model Precision Over Epochs')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# Compute detailed metrics for the validation set
print("\nValidation Set Performance:")
val_auc = roc_auc_score(Y_val, Y_val_pred)
val_conf_matrix = confusion_matrix(Y_val, Y_val_pred_bin)
val_class_report = classification_report(Y_val, Y_val_pred_bin)
val_balanced_acc = balanced_accuracy_score(Y_val, Y_val_pred_bin)

print("AUC Score:", val_auc)
print("Confusion Matrix:\n", val_conf_matrix)
print("Classification Report:\n", val_class_report)
print("Balanced Accuracy:", val_balanced_acc)

# Compute detailed metrics for the test set
print("\nTest Set Performance:")
test_auc = roc_auc_score(Y_test, Y_test_pred)
test_conf_matrix = confusion_matrix(Y_test, Y_test_pred_bin)
test_class_report = classification_report(Y_test, Y_test_pred_bin)
test_balanced_acc = balanced_accuracy_score(Y_test, Y_test_pred_bin)

print("AUC Score:", test_auc)
print("Confusion Matrix:\n", test_conf_matrix)
print("Classification Report:\n", test_class_report)
print("Balanced Accuracy:", test_balanced_acc)

