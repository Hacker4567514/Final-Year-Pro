from flask import Flask, request, jsonify, render_template
import pandas as pd
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
CORS(app)

# Load data and train model
file_path = r'C:\Users\Rohit\OneDrive\Documents\b-tech-projecct--main\b-tech-projecct--main\Business_Info_12000_Rows.xlsx'
data = pd.read_excel(file_path)

# Define "high potential" threshold (top 25% in rating and reviews)
rating_threshold = data['Rating'].quantile(0.75)
reviews_threshold = data['Number of Reviews'].quantile(0.75)

# Create target variable 'High Potential': 1 if both conditions are met, else 0
data['High_Potential'] = ((data['Rating'] >= rating_threshold) & 
                          (data['Number of Reviews'] >= reviews_threshold)).astype(int)

# Select features and target
X = data[['Location', 'Sector', 'Business Stage', 'Category', 'Rating', 'Number of Reviews']]
y = data['High_Potential']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess categorical features
categorical_features = ['Location', 'Sector', 'Business Stage', 'Category']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", report)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract filter criteria from the POST request
    filters = request.json
    location = filters.get("location")
    sector = filters.get("sector")
    stage = filters.get("stage")
    category = filters.get("category")
    
    # Filter the dataset based on the input criteria
    filtered_data = data.copy()
    if location:
        filtered_data = filtered_data[filtered_data['Location'] == location]
    if sector:
        filtered_data = filtered_data[filtered_data['Sector'] == sector]
    if stage:
        filtered_data = filtered_data[filtered_data['Business Stage'] == stage]
    if category:
        filtered_data = filtered_data[filtered_data['Category'] == category]
    
    # If no results match the filters, return an empty list
    if filtered_data.empty:
        return jsonify(results=[])

    # Prepare the filtered data for prediction by the model
    predictions = pipeline.predict(filtered_data[['Location', 'Sector', 'Business Stage', 'Category', 'Rating', 'Number of Reviews']])
    filtered_data['Prediction'] = predictions
    
    # Sort by 'High Potential' status, 'Rating', and 'Number of Reviews'
    sorted_data = filtered_data.sort_values(by=['Prediction', 'Rating', 'Number of Reviews'], ascending=[False, False, False])
    
    # Select relevant columns and convert to a list of dictionaries for JSON response
    results = sorted_data[['Name','Rating', 'Number of Reviews', 'Prediction']].to_dict(orient='records')
    
    return jsonify(results=results)

# New endpoint to get unique options for dropdowns
@app.route('/options', methods=['GET'])
def get_options():
    options = {
        'locations': data['Location'].dropna().unique().tolist(),
        'sectors': data['Sector'].dropna().unique().tolist(),
        'stages': data['Business Stage'].dropna().unique().tolist(),
        'categories': data['Category'].dropna().unique().tolist()
    }
    return jsonify(options)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
