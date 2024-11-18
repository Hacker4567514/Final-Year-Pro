import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
file_path = r'C:\Users\Harshal\Downloads\B-TECH-PROJECT-main (1)\B-TECH-PROJECT-main\B-TECH PROJECT\b-tech-projecct--main\Business_Info_12000_Rows.xlsx'
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

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", report)