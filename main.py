# Importing all the libraries needed for implementation.
import csv
import joblib
import json
import nltk
import numpy as np
import pandas as pd
import re
import scipy.sparse as sp
import string

from fastapi import FastAPI, File, UploadFile, HTTPException
from nltk.stem.porter import PorterStemmer
from pydantic import BaseModel, ValidationError
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from mangum import Mangum

# Initializing the app constructor.
app = FastAPI()

# Defining the structure of single item prediction.
class Item(BaseModel):
    attachment_count: str = None
    attachment_extension: str = None
    email_from: str
    email_subject: str
    closeNotes: str

# Function that review if the file has a csv extension.
def is_csv(file):
    return file.filename.endswith(".csv")

# Function that review if the file has a json extension.
def is_json(file):
    return file.filename.endswith(".json")

# Function that validate tjat the csv has the correct/expected format.
def validate_csv_contents(csv_rows):
    # Perform content validation here
    if len(csv_rows) < 2:
        raise ValidationError("CSV file must have at least two rows (headers and data)")
    headers = csv_rows[0]
    data_rows = csv_rows[1:]
    if len(headers) != len(data_rows[0]):
        raise ValidationError("Number of columns in data rows must match number of headers")

# Function that uses the model to obtain the predictions and return the information.
def obtain_predictions(model, data):
    predictions = model.predict(data)
    print(predictions)
    probabilities = model.predict_proba(data)
    print(probabilities)

    return predictions, probabilities

# Function that uses relevant information such as model, prediction and probabilites to construct endpoint response.
def generate_response(model_name, preds, probs):
    # Define lambda function to apply the logic
    category_lambda = lambda x: "Not Spam" if x == 0 else "Spam"
    
    # Apply lambda function to each value in preds using list comprehension
    categories = [category_lambda(pred) for pred in preds]
    
    # Construct the response dictionary
    response_data = {
        "model" : model_name,
        "predicted_category": categories,
        "predictions": preds.tolist(),
        "probabilities": probs.tolist()
    }
    # Convert the response dictionary to JSON format
    json_response = json.dumps(response_data)

    return json_response

# Function to transform Email Subject - Pipeline 1st step.
def get_importantFeatures(sent):
    sent = sent.lower()
    
    returnList = []
    sent = nltk.word_tokenize(sent)
    for i in sent:
        if i.isalnum():
            returnList.append(i)
            
    return returnList

# Function to transform Email Subject - Pipeline 2nd step.
def removing_stopWords(sent):
    returnList = []
    for i in sent:
        if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
            returnList.append(i)
    return returnList

# Function to transform Email Subject - Pipeline 3rd step.
def potter_stem(sent):
    ps = PorterStemmer()
    returnList = []
    for i in sent:
        returnList.append(ps.stem(i))
    return " ".join(returnList)

# Function to extract only email from email field.
def extract_email(text):
    # Define a regular expression pattern to match email addresses
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    # Use findall() to find all matches of the pattern in the text
    matches = re.findall(pattern, text)    
    return matches

# Function to extract only the domain of an email direction.
def extractDomains(data, name):
    extract_domain = lambda email: email.split('@')[-1].split('>')[0].strip() if isinstance(email, str) else ''
    data[name] = data[name].apply(extract_domain)

# Function to encoded target classes.
def isSpamClassification(row):
    classification = row["closeNotes"].lower()
    if classification == "spam":
        return 1
    else:
        return 0

# Function that contain the complete pipeline to pre-process the input data.
def preprocess_data(df):
    # Creating constructors for specific workers.
    # On manipulating data applying encoding, creating vectors and (TODO Porter Stemmer)
    label_encoder = LabelEncoder()

    # Applying encoding in order to group by category.
    df["Encoded_Target"] = df.apply(isSpamClassification, axis=1)  # Assuming isSpamClassification is defined elsewhere.

    # Drop rows without email subject and re-index dataset.
    df.dropna(subset=['Email Subject'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Filling null values with 0 in 'Attachment Count'.
    df['Attachment Count'] = df['Attachment Count'].fillna(0)

    # Eliminate string values in 'Attachment Count', group by single key, and re-index dataset.
    df = df[pd.to_numeric(df['Attachment Count'], errors='coerce').notnull()]
    df['Attachment Count'] = df['Attachment Count'].astype(float)
    df['Attachment Count'] = df['Attachment Count'].astype(int)
    df.reset_index(drop=True, inplace=True)

    # Imputation to email's subject.
    df['imp_feature'] = df['Email Subject'].apply(get_importantFeatures)
    df['imp_feature'] = df['imp_feature'].apply(removing_stopWords)
    df['imp_feature'] = df['imp_feature'].apply(potter_stem)

    # Extract email from the raw data and plain the vector obtained to only save strings.
    df["Senders_Email"] = df['Email From'].apply(lambda x: extract_email(x))
    df["Senders_Email"] = df["Senders_Email"].apply(lambda x: ', '.join(x))

    # Extract domain from the senders email.
    df["Email_domain"] = df["Senders_Email"]
    extractDomains(df, "Email_domain")

    # Encode the domain value, in order to see how many emails had been sent by domain.
    label_encoder = LabelEncoder()
    df["Email_domain_Encoded"] = label_encoder.fit_transform(df["Email_domain"])

    # Encode the complete email value, in order to see how many emails had been sent by email direction.
    df["Email_sender_Encoded"] = label_encoder.fit_transform(df["Senders_Email"])

    # Extract length of email domain from 'Email From'.
    df['Email_domain_Length'] = df['Email_domain'].apply(len)

    # Obtain the length of the email's subject.
    df['Subject_Length'] = df['Email Subject'].apply(len)

    # Define a list of keywords that might indicate spam.
    spam_keywords = ['prize', 'lottery', 'win', 'free', 'money', 'offer', 'urgent']

    # Check for the presence of any spam keyword in the subject line.
    df['Subject_Contains_SpamKeyword'] = df['Email Subject'].apply(lambda x: any(keyword in x.lower() for keyword in spam_keywords)).astype(int)
    
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    # Creating our testing dataset.
    X_text = df['imp_feature']
    X_text_tfidf = tfidf_vectorizer.transform(X_text)

    # Drop unused columns.
    df.drop(columns=['Attachment Extension', 'Email From', 'Email Subject', 'closeNotes', 'Email_domain', 'Senders_Email', 'imp_feature', 'Encoded_Target'], inplace=True)

    # Convert other features to sparse matrix.
    X_test_sparse = sp.csr_matrix(df.values)

    # Combine sparse matrices.
    X_combined = sp.hstack([X_test_sparse, X_text_tfidf], format='csr')
    
    # Print the shape of X_combined
    print("Shape of X_combined:", X_combined.shape)

    return X_combined

# Function that calls and generated the prediction based on input and model.
def predict_file(df, model_type: str):
    # Call the preprocess_data function
    processed_data = preprocess_data(df)
    # Check if processed_data is empty
    if processed_data.nnz == 0:
        return "Processed data is empty."

    # Load the best model based on model_type
    if model_type == "tree":
        best_model_class = joblib.load('best_tree_model.pkl')
    elif model_type == "rf":
        best_model_class = joblib.load('best_rf_model.pkl')
    elif model_type == "naive":
        best_model_class = joblib.load('best_naive_model.pkl')
    elif model_type == "log_reg":
        best_model_class = joblib.load('best_log_reg_model.pkl')
    elif model_type == "knn":
        best_model_class = joblib.load('best_knn_model.pkl')
    elif model_type == "svm":
        best_model_class = joblib.load('best_svm_model.pkl')
    elif model_type == "nn":
        best_model_class = joblib.load('best_nn_model.pkl')
    else:
        raise ValueError("Invalid model type")

    # Get selected parameters
    selected_params = best_model_class.best_params_
    print(f'Selected Parameters for {model_type.capitalize()}: {selected_params}')

    # Obtain predictions
    prediction, probabilities = obtain_predictions(best_model_class, processed_data)

    # Generate response
    api_response = generate_response(f"{model_type.capitalize()} Classifier", prediction, probabilities)

    return api_response

# Root endpoint when initialized the FastAPI project.
@app.get("/")
def read_root(name: str = "World"):
    return {"message": f"Hello, {name}!"}

# Endpoint used to make single item prediction for Random Forest Algorithm.
@app.post("/rf_single_prediction/")
def create_item(item: Item):
    message = f"Analyzing an email from {item.email_from} classified as {item.closeNotes}"
    return {"message": message}

# Endpoint used to make file predictions for Random Forest Algorithm.
@app.post("/rf_file_prediction/")
async def rf_upload_file(file: UploadFile = File(...)):
    if is_csv(file):
        try:
            contents = await file.read()
            csv_data = csv.reader(contents.decode().splitlines())
            csv_rows = list(csv_data)
            validate_csv_contents(csv_rows)
            # Convert CSV data to DataFrame
            df = pd.DataFrame(csv_rows[1:], columns=csv_rows[0])
            return {"file_type": "csv", "filename": file.filename, "processed_message": predict_file(df, "rf")}
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif is_json(file):
        try:
            contents = await file.read()
            json_data = json.loads(contents.decode())
            items = json_data['items']
            # Initialize an empty list to store dictionaries
            data_list = []
            # Iterate over each item and append it to the list
            for item in items:
                data_list.append({
                    "Attachment Count": item["Attachment Count"][0],
                    "Attachment Extension": item["Attachment Extension"][0],
                    "Email From": item["Email From"][0],
                    "Email Subject": item["Email Subject"][0],
                    "closeNotes": item["closeNotes"][0]
                })
            # Convert JSON data to DataFrame
            df = pd.DataFrame(data_list)
            return {"file_type": "json", "filename": file.filename, "processed_message": predict_file(df, "rf")}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
    else:
        raise HTTPException(status_code=400, detail="Only CSV and JSON files are allowed")
    
# Endpoint used to make file predictions for Decision Tree Algorithm.
@app.post("/tree_file_prediction/")
async def tree_upload_file(file: UploadFile = File(...)):
    if is_csv(file):
        try:
            contents = await file.read()
            csv_data = csv.reader(contents.decode().splitlines())
            csv_rows = list(csv_data)
            validate_csv_contents(csv_rows)
            # Convert CSV data to DataFrame
            df = pd.DataFrame(csv_rows[1:], columns=csv_rows[0])
            return {"file_type": "csv", "filename": file.filename, "processed_message": predict_file(df, "tree")}
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif is_json(file):
        try:
            contents = await file.read()
            json_data = json.loads(contents.decode())
            items = json_data['items']
            # Initialize an empty list to store dictionaries
            data_list = []
            # Iterate over each item and append it to the list
            for item in items:
                data_list.append({
                    "Attachment Count": item["Attachment Count"][0],
                    "Attachment Extension": item["Attachment Extension"][0],
                    "Email From": item["Email From"][0],
                    "Email Subject": item["Email Subject"][0],
                    "closeNotes": item["closeNotes"][0]
                })
            # Convert JSON data to DataFrame
            df = pd.DataFrame(data_list)
            return {"file_type": "json", "filename": file.filename, "processed_message": predict_file(df, "tree")}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
    else:
        raise HTTPException(status_code=400, detail="Only CSV and JSON files are allowed")
     
# Endpoint used to make file predictions for Logistic Regression Algorithm.
@app.post("/log_reg_file_prediction/")
async def log_reg_upload_file(file: UploadFile = File(...)):
    if is_csv(file):
        try:
            contents = await file.read()
            csv_data = csv.reader(contents.decode().splitlines())
            csv_rows = list(csv_data)
            validate_csv_contents(csv_rows)
            # Convert CSV data to DataFrame
            df = pd.DataFrame(csv_rows[1:], columns=csv_rows[0])
            return {"file_type": "csv", "filename": file.filename, "processed_message": predict_file(df, "log_reg")}
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif is_json(file):
        try:
            contents = await file.read()
            json_data = json.loads(contents.decode())
            items = json_data['items']
            # Initialize an empty list to store dictionaries
            data_list = []
            # Iterate over each item and append it to the list
            for item in items:
                data_list.append({
                    "Attachment Count": item["Attachment Count"][0],
                    "Attachment Extension": item["Attachment Extension"][0],
                    "Email From": item["Email From"][0],
                    "Email Subject": item["Email Subject"][0],
                    "closeNotes": item["closeNotes"][0]
                })
            # Convert JSON data to DataFrame
            df = pd.DataFrame(data_list)
            return {"file_type": "json", "filename": file.filename, "processed_message": predict_file(df, "log_reg")}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
    else:
        raise HTTPException(status_code=400, detail="Only CSV and JSON files are allowed")

# Endpoint used to make file predictions for Naive Bayes Algorithm.
@app.post("/naive_file_prediction/")
async def naive_upload_file(file: UploadFile = File(...)):
    if is_csv(file):
        try:
            contents = await file.read()
            csv_data = csv.reader(contents.decode().splitlines())
            csv_rows = list(csv_data)
            validate_csv_contents(csv_rows)
            # Convert CSV data to DataFrame
            df = pd.DataFrame(csv_rows[1:], columns=csv_rows[0])
            return {"file_type": "csv", "filename": file.filename, "processed_message": predict_file(df, "naive")}
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif is_json(file):
        try:
            contents = await file.read()
            json_data = json.loads(contents.decode())
            items = json_data['items']
            # Initialize an empty list to store dictionaries
            data_list = []
            # Iterate over each item and append it to the list
            for item in items:
                data_list.append({
                    "Attachment Count": item["Attachment Count"][0],
                    "Attachment Extension": item["Attachment Extension"][0],
                    "Email From": item["Email From"][0],
                    "Email Subject": item["Email Subject"][0],
                    "closeNotes": item["closeNotes"][0]
                })
            # Convert JSON data to DataFrame
            df = pd.DataFrame(data_list)
            return {"file_type": "json", "filename": file.filename, "processed_message": predict_file(df, "naive")}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
    else:
        raise HTTPException(status_code=400, detail="Only CSV and JSON files are allowed")

# Endpoint used to make file predictions for K-Nearest Neighbour Algorithm.
@app.post("/knn_file_prediction/")
async def knn_upload_file(file: UploadFile = File(...)):
    if is_csv(file):
        try:
            contents = await file.read()
            csv_data = csv.reader(contents.decode().splitlines())
            csv_rows = list(csv_data)
            validate_csv_contents(csv_rows)
            # Convert CSV data to DataFrame
            df = pd.DataFrame(csv_rows[1:], columns=csv_rows[0])
            return {"file_type": "csv", "filename": file.filename, "processed_message": predict_file(df, "knn")}
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif is_json(file):
        try:
            contents = await file.read()
            json_data = json.loads(contents.decode())
            items = json_data['items']
            # Initialize an empty list to store dictionaries
            data_list = []
            # Iterate over each item and append it to the list
            for item in items:
                data_list.append({
                    "Attachment Count": item["Attachment Count"][0],
                    "Attachment Extension": item["Attachment Extension"][0],
                    "Email From": item["Email From"][0],
                    "Email Subject": item["Email Subject"][0],
                    "closeNotes": item["closeNotes"][0]
                })
            # Convert JSON data to DataFrame
            df = pd.DataFrame(data_list)
            return {"file_type": "json", "filename": file.filename, "processed_message": predict_file(df, "knn")}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
    else:
        raise HTTPException(status_code=400, detail="Only CSV and JSON files are allowed")

# Endpoint used to make file predictions for Support Vector Machine Algorithm.
@app.post("/svm_file_prediction/")
async def svm_upload_file(file: UploadFile = File(...)):
    if is_csv(file):
        try:
            contents = await file.read()
            csv_data = csv.reader(contents.decode().splitlines())
            csv_rows = list(csv_data)
            validate_csv_contents(csv_rows)
            # Convert CSV data to DataFrame
            df = pd.DataFrame(csv_rows[1:], columns=csv_rows[0])
            return {"file_type": "csv", "filename": file.filename, "processed_message": predict_file(df, "svm")}
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif is_json(file):
        try:
            contents = await file.read()
            json_data = json.loads(contents.decode())
            items = json_data['items']
            # Initialize an empty list to store dictionaries
            data_list = []
            # Iterate over each item and append it to the list
            for item in items:
                data_list.append({
                    "Attachment Count": item["Attachment Count"][0],
                    "Attachment Extension": item["Attachment Extension"][0],
                    "Email From": item["Email From"][0],
                    "Email Subject": item["Email Subject"][0],
                    "closeNotes": item["closeNotes"][0]
                })
            # Convert JSON data to DataFrame
            df = pd.DataFrame(data_list)
            return {"file_type": "json", "filename": file.filename, "processed_message": predict_file(df, "svm")}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
    else:
        raise HTTPException(status_code=400, detail="Only CSV and JSON files are allowed")

# Endpoint used to make file predictions for Neural Networks Algorithm.
@app.post("/neural_networks_file_prediction/")
async def neural_networks_upload_file(file: UploadFile = File(...)):
    if is_csv(file):
        try:
            contents = await file.read()
            csv_data = csv.reader(contents.decode().splitlines())
            csv_rows = list(csv_data)
            validate_csv_contents(csv_rows)
            # Convert CSV data to DataFrame
            df = pd.DataFrame(csv_rows[1:], columns=csv_rows[0])
            return {"file_type": "csv", "filename": file.filename, "processed_message": predict_file(df, "nn")}
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif is_json(file):
        try:
            contents = await file.read()
            json_data = json.loads(contents.decode())
            items = json_data['items']
            # Initialize an empty list to store dictionaries
            data_list = []
            # Iterate over each item and append it to the list
            for item in items:
                data_list.append({
                    "Attachment Count": item["Attachment Count"][0],
                    "Attachment Extension": item["Attachment Extension"][0],
                    "Email From": item["Email From"][0],
                    "Email Subject": item["Email Subject"][0],
                    "closeNotes": item["closeNotes"][0]
                })
            # Convert JSON data to DataFrame
            df = pd.DataFrame(data_list)
            return {"file_type": "json", "filename": file.filename, "processed_message": predict_file(df, "nn")}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
    else:
        raise HTTPException(status_code=400, detail="Only CSV and JSON files are allowed")

handler = Mangup(app)