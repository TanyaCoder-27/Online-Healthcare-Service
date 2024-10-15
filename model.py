'''

import pandas as pd
import joblib
import pickle
# Load the dataset
df = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/NEW MINI PROJECT/DiseaseAndSymptoms.csv')

# Display the first few rows of the dataset
print(df.head())

# Fill missing values with a placeholder (e.g., 'Unknown')
df.fillna('Unknown', inplace=True)

# Alternatively, you can drop rows with missing values
# df.dropna(inplace=True)

from sklearn.preprocessing import OneHotEncoder

# Select symptom columns
symptom_columns = [col for col in df.columns if 'Symptom' in col]

# Apply one-hot encoding
encoder = OneHotEncoder()
encoded_symptoms = encoder.fit_transform(df[symptom_columns])

# Convert to DataFrame
encoded_symptoms_df = pd.DataFrame(encoded_symptoms.toarray(), columns=encoder.get_feature_names_out(symptom_columns))

# Concatenate encoded symptoms with the original DataFrame
df_encoded = pd.concat([df, encoded_symptoms_df], axis=1)

# Drop original symptom columns
df_encoded.drop(symptom_columns, axis=1, inplace=True)

# Features (encoded symptoms)
X = df_encoded.drop('Disease', axis=1)

# Target variable (disease)
y = df_encoded['Disease']

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

#joblib.dump(svm_model, 'svm_model.pkl')
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)

'''

'''
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import pickle

# Load your dataset
symptoms_df = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/NEW MINI PROJECT/DiseaseAndSymptoms.csv')

# Assuming your dataset has a column for each symptom
symptom_columns = ['Symptom_1', 'Symptom_2', 'Symptom_3','Symptom_4', 'Symptom_5', 'Symptom_6','Symptom_7', 'Symptom_8', 'Symptom_9','Symptom_10', 'Symptom_11', 'Symptom_12','Symptom_13', 'Symptom_14', 'Symptom_15','Symptom_16', 'Symptom_17']  # Adjust based on your dataset

# Flatten the symptom columns to get a list of all symptoms
all_symptoms = symptoms_df[symptom_columns].values.flatten()

# Initialize and fit the LabelEncoder
encoder = LabelEncoder()
encoder.fit(all_symptoms)

# Encode the symptoms in the dataset
for col in symptom_columns:
    symptoms_df[col] = encoder.transform(symptoms_df[col])

# Prepare your features and labels
X = symptoms_df[symptom_columns]
y = symptoms_df['Disease']

# Train your SVM model
model = svm.SVC()
model.fit(X, y)

# Save the model and the encoder
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)
'''


'''
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/NEW MINI PROJECT/DiseaseAndSymptoms.csv')

# Fill missing values with a placeholder (e.g., 'Unknown')
df.fillna('Unknown', inplace=True)

# Select symptom columns
symptom_columns = [col for col in df.columns if 'Symptom' in col]

# Flatten the symptom columns to get a list of all symptoms
all_symptoms = df[symptom_columns].values.flatten()

# Initialize and fit the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(all_symptoms)

# Encode the symptoms in the dataset
for col in symptom_columns:
    df[col] = label_encoder.transform(df[col])

# Features (encoded symptoms)
X = df[symptom_columns]

# Target variable (disease)
y = df['Disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model and the label encoder
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)

with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)


'''

#-------------------------------------------------------

import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset with the correct encoding
df = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/NEW MINI PROJECT/New_Disease_Symptoms.csv', encoding='latin1')

# Fill missing values with a placeholder (e.g., 'Unknown')
df.fillna('Unknown', inplace=True)

# Split the symptoms into lists
df['Symptoms'] = df['Symptoms'].apply(lambda x: x.split(','))

# Use MultiLabelBinarizer to encode the symptoms
mlb = MultiLabelBinarizer()
symptoms_encoded = mlb.fit_transform(df['Symptoms'])

# Features (encoded symptoms)
X = symptoms_encoded

# Target variable (disease)
y = df['Disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model and the MultiLabelBinarizer
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)

with open('mlb.pkl', 'wb') as file:
    pickle.dump(mlb, file)
