import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib  # For saving and loading the model

# Function to cap outliers in the data
def cap_outliers_array(X):
    X_capped = X.copy()
    for i in range(X_capped.shape[1]):
        col = X_capped[:, i]
        Q1 = np.percentile(col, 25)
        Q3 = np.percentile(col, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X_capped[:, i] = np.clip(col, lower_bound, upper_bound)
    return X_capped

# After defining the function, load the model
best_model = joblib.load('model_pipeline (2).pkl')

# Set up Streamlit interface
st.set_page_config(page_title="Customer Satisfaction Prediction", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.4)), 
                    url("https://wallpapers.com/images/hd/sunset-silhouette-airplane-brh2gmlmjhnj74dv.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: #FFFFFF;
    }
    h1{
        color: #FFFFFF !important;
    }
    .st-bq, .st-cy, .st-co {
        color: #000000 !important;
    }
    .stSidebar {
        background-color: rgba(255, 255, 255, 0.4) !important;
        font-size: 25px !important;
        font-weight: bold !important;
        color: black !important;
    }
    .stButton>button {
        color: black;
        font-weight: bold;
        font-size: 20px;
        cursor: pointer;
        text-align: center;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .stSelectbox, .stNumberInput, .stTextInput, .stDataFrame {
        background-color: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 10px;
    }
    .stForm {
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    hr {
        border-top: 1px solid rgba(255, 255, 255, 0.8) !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Style for the button inside the form */
    .stForm button {
        color: black;
        font-weight: bold;
        font-size: 20px;
        cursor: pointer;
        text-align: center;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Airline Customer Satisfaction Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Choose input method
st.sidebar.title("Input Method")
option = st.sidebar.radio("Choose input method:", ['Upload CSV File', 'Manual Input'])
uploaded_file = None

# Upload CSV file
if option == 'Upload CSV File':
    st.subheader("File Input")
    st.markdown("""
        <div style="background-color: rgba(169, 169, 169, 0.2); color: #FBDB93; padding: 10px; border-radius: 5px;">
          <strong>Upload your CSV file:</strong>
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.success("✅ File uploaded successfully!")
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head(), use_container_width=True)
        df.drop(columns=['id'], inplace=True)
        # Ensure the 'satisfaction' column is not present in the uploaded data
        if 'satisfaction' in df.columns:
            df.drop('satisfaction', axis=1, inplace=True)

        # Use only the input columns that the model was trained on
        X_input = df

        # Display the Predict button after uploading data
        predict_button = st.button("Predict")

        if predict_button:
            # Use the model to predict 'satisfaction'
            y_result = best_model.predict(X_input)
        
            # Convert the predictions from 0 or 1 to text labels
            satisfaction_label = y_result
        
            # Display the result
            st.success("✅ Prediction Result:")
            st.write("Predicted Satisfaction Levels: ", satisfaction_label)
    except Exception as e:
        st.error(f"❌ Error loading CSV file: {e}")

# Manual input
elif option == 'Manual Input':
    st.subheader("Manual Input")
    st.markdown("""
        <div style="background-color: rgba(169, 169, 169, 0.2); color: #FBDB93; padding: 10px; border-radius: 5px;">
          <strong>Please input values for the following flight features:</strong>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    input_data = {}

    features = [
        ('id', 'number_input', [0, 10000, 1]),
        ('Gender', 'selectbox', ['Male', 'Female']),
        ('Customer Type', 'selectbox', ['Loyal Customer', 'Disloyal Customer']),
        ('Age', 'number_input', [1, 100, 30]),
        ('Type of Travel', 'selectbox', ['Personal Travel', 'Business Travel']),
        ('Class', 'selectbox', ['Eco', 'Eco Plus', 'Business']),
        ('Flight Distance', 'number_input', [0, 10000, 500]),
        ('Inflight wifi service', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('Departure/Arrival time convenient', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('Ease of Online booking', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('Gate location', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('Food and drink', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('Online boarding', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('Seat comfort', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('Inflight entertainment', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('On-board service', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('Leg room service', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('Baggage handling', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('Checkin service', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('Inflight service', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('Cleanliness', 'selectbox', [0, 1, 2, 3, 4, 5]),
        ('Departure Delay in Minutes', 'number_input', [0, 2000, 0]),
        ('Arrival Delay in Minutes', 'number_input', [0, 2000, 0])
    ]

    with st.form("manual_input_form"):
        for i in range(0, len(features), 3):
            cols = st.columns(3)
            for j, (feature, input_type, options) in enumerate(features[i:i+3]):
                with cols[j]:
                    if input_type == 'selectbox':
                        input_data[feature] = st.selectbox(f"**{feature}:**", options, key=feature)
                    elif input_type == 'number_input':
                        input_data[feature] = st.number_input(
                            f"**{feature}:**", min_value=options[0], max_value=options[1], value=options[2], key=feature
                        )
        st.markdown("---")
        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([input_data])
        input_df.drop(columns=['id'], inplace=True)
        # Use the model to predict
        y_pred = best_model.predict(input_df)
    
        # Convert the prediction from 0 or 1 to text labels
        satisfaction_label = y_pred[0]
    
        # Display the result
        st.success("✅ Prediction Result:")
        st.write("Predicted Satisfaction Level: ", satisfaction_label)
