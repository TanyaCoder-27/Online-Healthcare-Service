#'C:/Users/tanya/OneDrive/Desktop/NEW MINI PROJECT/Disease precaution.csv'



'''
from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the enhanced dataset
#disease_info_df = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/NEW MINI PROJECT/Disease_precaution.csv')
disease_info_df = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/NEW MINI PROJECT/Disease_precaution.csv', encoding='ISO-8859-1')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptom1 = request.form['symptom1']
        symptom2 = request.form['symptom2']
        symptom3 = request.form['symptom3']
        
        # Preprocess the input symptoms and make a prediction
        symptoms = [symptom1, symptom2, symptom3]
        prediction = model.predict([symptoms])
        
        # Fetch additional information based on the prediction
        disease_info = disease_info_df[disease_info_df['Disease'] == prediction[0]].iloc[0]
        
        # Prepare the information to be displayed
        disease_details = {
            'Disease': disease_info['Disease'],
            'Description': disease_info['Description'],
            'Precautions': '; '.join([disease_info['Precaution_1'], disease_info['Precaution_2'], disease_info['Precaution_3'], disease_info['Precaution_4']]),
            'Food to Take': disease_info['Food_to_take'],
            'Food to Avoid': disease_info['Food_to_avoid']
        }
        
        return render_template('predict.html', prediction=prediction[0], info=disease_details)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)

'''
'''
from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Load the enhanced dataset with specified encoding
disease_info_df = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/NEW MINI PROJECT/Disease_precaution.csv', encoding='ISO-8859-1')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptom1 = request.form['symptom1']
        symptom2 = request.form['symptom2']
        symptom3 = request.form['symptom3']
        
        # Preprocess the input symptoms
        symptoms = [symptom1, symptom2, symptom3]
        
        # Handle unseen labels by mapping them to '<unknown>'
        symptoms = [s if s in label_encoder.classes_ else '<unknown>' for s in symptoms]
        
        # Add '<unknown>' to classes and transform
        label_encoder.classes_ = np.append(label_encoder.classes_, '<unknown>')
        encoded_symptoms = label_encoder.transform(symptoms)
        
        # Ensure the input has 17 features (padding with zeros if necessary)
        input_features = np.zeros((1, 17))
        input_features[0, :len(encoded_symptoms)] = encoded_symptoms
        
        # Make a prediction
        prediction = model.predict(input_features)
        
        # Fetch additional information based on the prediction
        disease_info = disease_info_df[disease_info_df['Disease'] == prediction[0]].iloc[0]
        
        # Prepare the information to be displayed
        disease_details = {
            'Disease': disease_info['Disease'],
            'Description': disease_info['Description'],
            'Precautions': '; '.join([disease_info['Precaution_1'], disease_info['Precaution_2'], disease_info['Precaution_3'], disease_info['Precaution_4']]),
            'Food to Take': disease_info['Food_to_take'],
            'Food to Avoid': disease_info['Food_to_avoid']
        }
        
        return render_template('predict.html', prediction=prediction[0], info=disease_details)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
'''


from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Load the MultiLabelBinarizer
with open('mlb.pkl', 'rb') as file:
    mlb = pickle.load(file)

# Load the enhanced dataset for output with specified encoding
disease_info_df = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/NEW MINI PROJECT/New_Disease_Info.csv', encoding='ISO-8859-1')

@app.route('/')
def home():
    return render_template('index.html')
'''
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get symptoms from the form input
        symptom1 = request.form['symptom1']
        symptom2 = request.form['symptom2']
        symptom3 = request.form['symptom3']
        
        # Preprocess the input symptoms
        symptoms = [symptom1, symptom2, symptom3]
        symptoms_encoded = mlb.transform([symptoms])  # Encode the symptoms using the trained MultiLabelBinarizer

        # Ensure the input has the same number of features as the model expects
        input_features = np.zeros((1, len(mlb.classes_)))
        input_features[0, :symptoms_encoded.shape[1]] = symptoms_encoded

        # Make a prediction using the SVM model
        prediction = svm_model.predict(input_features)
        predicted_disease = prediction[0]

        # Fetch additional information from the dataset based on the prediction
        disease_info = disease_info_df[disease_info_df['Disease'] == predicted_disease].iloc[0]
        
        # Prepare the information to be displayed
        disease_details = {
            'Disease': disease_info['Disease'],
            'Description': disease_info['Description'],
            'Precautions': '; '.join([str(disease_info['Precaution_1']), str(disease_info['Precaution_2']), 
                                      str(disease_info['Precaution_3']), str(disease_info['Precaution_4'])]),
            'Food to Take': disease_info['Foods to take'],
            'Food to Avoid': disease_info['Foods to avoid']
        }

        # Render the predict page with the prediction and additional information
        return render_template('predict.html', prediction=predicted_disease, info=disease_details)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
'''


'''
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get symptoms from the form input
        symptom1 = request.form['symptom1']
        symptom2 = request.form['symptom2']
        symptom3 = request.form['symptom3']
        
        # Preprocess the input symptoms
        symptoms = [symptom1, symptom2, symptom3]
        symptoms_encoded = mlb.transform([symptoms])  # Encode the symptoms using the trained MultiLabelBinarizer

        # Ensure the input has the same number of features as the model expects
        input_features = np.zeros((1, len(mlb.classes_)))
        input_features[0, :symptoms_encoded.shape[1]] = symptoms_encoded

        # Make a prediction using the SVM model
        prediction = svm_model.predict(input_features)
        predicted_disease = prediction[0]

        # Fetch additional information from the dataset based on the prediction
        disease_info = disease_info_df[disease_info_df['Disease'] == predicted_disease].iloc[0]

        # Print column names to troubleshoot the issue
        print(disease_info_df.columns)  # Check if 'Precaution_1' exists or if the columns are named differently

        # Handle the case where column names are different
        try:
            precautions = '; '.join([str(disease_info['Precaution_1']), str(disease_info['Precaution_2']), 
                                      str(disease_info['Precaution_3']), str(disease_info['Precaution_4'])])
        except KeyError:
            precautions = 'Precaution details not available'
        
        # Prepare the information to be displayed
        disease_details = {
            'Disease': disease_info['Disease'],
            'Description': disease_info.get('Description', 'Description not available'),
            'Precautions': disease_info.get('Precautions','Precautions not available'),
            'Food to Take': disease_info.get('Foods to take', 'Food to take information not available'),
            'Food to Avoid': disease_info.get('Foods to avoid', 'Food to avoid information not available')
        }

        # Render the predict page with the prediction and additional information
        return render_template('predict.html', prediction=predicted_disease, info=disease_details)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
'''

from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Load the MultiLabelBinarizer
with open('mlb.pkl', 'rb') as file:
    mlb = pickle.load(file)

# Load the enhanced dataset for output with specified encoding
disease_info_df = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/NEW MINI PROJECT/New_Disease_Info.csv', encoding='ISO-8859-1')

# List of all possible symptoms for the user to choose from
all_symptoms = mlb.classes_

from transformers import pipeline

# Load a pre-trained model for token classification (NER)
nlp = pipeline("ner")

# Example: Extract symptoms from a sentence
def extract_symptoms(user_input):
    entities = nlp(user_input)
    symptoms = [entity['word'] for entity in entities if entity['entity'].startswith('B')]  # Assuming B- for beginning of symptom
    return symptoms



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input sentence
        user_input = request.form['user_input']
        
        # Use Hugging Face model to extract symptoms from the sentence
        symptoms = extract_symptoms(user_input)

        # Check if we extracted symptoms, otherwise return an error
        if not symptoms:
            return "No symptoms detected. Please try again."

        # Preprocess the extracted symptoms for the model
        symptoms_encoded = mlb.transform([symptoms])

        # Make a prediction using the SVM model
        prediction = svm_model.predict(symptoms_encoded)
        predicted_disease = prediction[0]

        # Fetch additional information from the dataset
        disease_info = disease_info_df[disease_info_df['Disease'] == predicted_disease].iloc[0]

        # Prepare the information to be displayed
        disease_details = {
            'Disease': disease_info['Disease'],
            'Description': disease_info.get('Description', 'Description not available'),
            'Precautions': '; '.join(filter(None, [str(disease_info.get(f'Precaution_{i}', '')) for i in range(1, 5)])),
            'Food to Take': disease_info.get('Foods to take', 'Food to take information not available'),
            'Food to Avoid': disease_info.get('Foods to avoid', 'Food to avoid information not available')
        }

        # Return the results in a conversational format
        return render_template('predict.html', prediction=predicted_disease, info=disease_details)


if __name__ == '__main__':
    app.run(debug=True)
