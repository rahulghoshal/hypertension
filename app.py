import streamlit as st
import pandas as pd
import joblib
#import mysql.connector

# Load the trained model and scaler
model = joblib.load('modelv1.pkl')
scaler = joblib.load('scalerv2.pkl')

st.title('Hypertension Prediction App')
st.write('Enter the patient information to predict the likelihood of hypertension and provide your details to record the visit.')

# Input fields for visitor's name and email
visitor_name = st.text_input('Your Name')
visitor_email = st.text_input('Your Email')


# Numerical features used in training
numerical_features = ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI']

# --- Collect inputs (same as you had) ---
age = st.slider('Age', 18, 90, 45)
salt_intake = st.slider('Salt Intake (grams/day)', 0.0, 20.0, 10.0, 0.1)
stress_score = st.slider('Stress Score (0-10)', 0, 10, 5)
sleep_duration = st.slider('Sleep Duration (hours)', 3.0, 10.0, 7.0, 0.1)
bmi = st.slider('BMI', 15.0, 40.0, 25.0, 0.1)

bp_history_display = st.selectbox('BP History', ['Normal', 'Prehypertension', 'Hypertension'])
family_history_display = st.selectbox('Family History of Hypertension', ['No', 'Yes'])
exercise_level_display = st.selectbox('Exercise Level', ['Low', 'Moderate', 'High'])
smoking_status_display = st.selectbox('Smoking Status', ['Non-Smoker', 'Smoker'])

medication_options = ['Unknown', 'ACE Inhibitor', 'Beta Blocker', 'Diuretic', 'Other']
medication_display = st.selectbox('Medication', medication_options)

# --- Map ordinal/binary features to numeric exactly as training did ---
bp_map = {'Normal': 0, 'Prehypertension': 1, 'Hypertension': 2}
exercise_map = {'Low': 0, 'Moderate': 1, 'High': 2}
binary_map_yesno = {'No': 0, 'Yes': 1}
smoking_map = {'Non-Smoker': 0, 'Smoker': 1}

# Scalar numeric values (raw)
raw_values = {
    'Age': age,
    'Salt_Intake': salt_intake,
    'Stress_Score': stress_score,
    'Sleep_Duration': sleep_duration,
    'BMI': bmi,
    # mapped single-column categorical values (matching training)
    'BP_History': bp_map[bp_history_display],
    'Exercise_Level': exercise_map[exercise_level_display],
    'Family_History': binary_map_yesno[family_history_display],
    'Smoking_Status': smoking_map[smoking_status_display]
}

# --- Build base input dataframe filled with zeros for all expected features ---
# Prefer to use model.feature_names_in_ so we guarantee the same columns used at training
if hasattr(model, 'feature_names_in_'):
    expected_features = list(model.feature_names_in_)
else:
    # fallback list — replace with the exact columns you used in training if needed
    expected_features = [
        'Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI',
        'Medication_Beta Blocker', 'Medication_Diuretic', 'Medication_Other',
        'Medication_Unknown', 'Medication_ACE Inhibitor',
        'BP_History', 'Exercise_Level', 'Family_History', 'Smoking_Status'
    ]

# Create a 1-row DataFrame with all zeros for expected features
input_df = pd.DataFrame([{c: 0 for c in expected_features}])

# --- Put raw numeric & mapped categorical single columns into df ---
for k, v in raw_values.items():
    if k in input_df.columns:
        input_df.at[0, k] = v
    else:
        # if the training features used different names (e.g., you had BP_History numeric),
        # try common alternative names or warn the developer
        st.warning(f"Warning: expected column '{k}' not found among model features.")

# --- Medication: set the dummy column(s) if they exist in expected_features ---
# Training used pd.get_dummies(..., drop_first=True), so one medication dummy may be missing.
# We'll create values for any medication column that exists in expected_features.
med_col_name = f"Medication_{medication_display}"
# handle exact match of names (your training used spaces e.g., 'Medication_Beta Blocker')
if med_col_name in input_df.columns:
    input_df.at[0, med_col_name] = 1
else:
    # Try a fallback transformation: sometimes training column names differ in spacing or case
    # Check all expected feature names and set the one that contains medication_display substring
    matched = [c for c in input_df.columns if c.startswith('Medication_') and medication_display.replace(' ', '_') in c.replace(' ', '_')]
    if matched:
        input_df.at[0, matched[0]] = 1
    else:
        # If no medication dummy matches, assume the dropped category was the baseline and nothing to set
        # (leave all medication dummies as 0)
        pass

# --- Scaling: IMPORTANT: use scaler.transform, NOT fit_transform ---
# Ensure numerical_features exist in input_df before scaling
num_cols_present = [c for c in numerical_features if c in input_df.columns]
if len(num_cols_present) != len(numerical_features):
    st.warning(f"Numerical features mismatch. Found: {num_cols_present}")

# scaler.transform expects a 2D array; input_df[num_cols_present] is already 1xN
input_df[num_cols_present] = scaler.transform(input_df[num_cols_present])

# --- Reindex to expected_features order to match training exactly ---
input_df = input_df[expected_features]


if st.button('Predict', key='predict_button_main'):
    # your code
    if not visitor_name or not visitor_email:
        st.warning("Please enter your Name and Email to proceed with the prediction.")
    else:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]
        # ---- SHOW OUTPUT ----
        if prediction[0] == 1:
            st.error(
                f"⚠️ High Risk: The model predicts a high likelihood of hypertension.\n"
                f"Probability: {prediction_proba[0]:.2f}"
            )
        else:
            st.success(
                f"✅ Low Risk: The model predicts low likelihood of hypertension.\n"
                f"Probability: {prediction_proba[0]:.2f}"
            )

        # Optional (debugging)
        #st.write("Processed Input DataFrame:")
        #st.dataframe(input_df)

# import mysql.connector

# # Function to save visitor data to MySQL
# def save_visitor_data(visitor_name, visitor_email, age, prediction):
#     try:
#         # Connect to MySQL
#         conn = mysql.connector.connect(
#             host="localhost",
#             user="root",           # For extra security, consider creating an INSERT-only user
#             password="rahulghoshal",
#             database="visitor_data"
#         )
#         cursor = conn.cursor()

#         sql = """
#         INSERT INTO visitors (name, email, age, prediction)
#         VALUES (%s, %s, %s, %s, %s, %s)
#         """
#         values = (visitor_name, visitor_email, age, prediction)

#         cursor.execute(sql, values)
#         # Commit the transaction
#         conn.commit()

#     except mysql.connector.Error as err:
#         print(f"Error: {err}")

    # finally:
    #     # Close the connection
    #     cursor.close()
    #     conn.close()
