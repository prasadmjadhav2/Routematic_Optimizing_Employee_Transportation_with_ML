import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load Dataset
file_path = 'routematic_dataset.csv'
df = pd.read_csv(file_path)

# Assuming 'df' is your DataFrame
features = [
    'Employee_ID', 'Distance_km', 'Duration_min', 'Shift_Type', 'Vehicle_ID',
    'Vehicle_Type', 'Fuel_Type', 'Preferred_Mode', 'Employee_Company',
    'Pickup_Address', 'Drop_Company', 'Drop_Address', 'Pickup_Time', 'Ride_Cost', 'Price_Revenue']

df_selected = df[features]

df_selected['Pickup_Time'] = df_selected['Pickup_Time'].str.extract(r"\b(\d{2}:\d{2})\b")

df_selected['Pickup_Time'] = df_selected['Pickup_Time'].str.strip().str.replace(r"^0|:00$", "", regex=True).str.replace(r":", ".", regex=False)

# Mapping dictionaries

shift_type_mapping = {
    'First Shift': '1',
    'General Shift': '2'
}

vehicle_type_mapping = {
    'Sedan': '1',
    'Electric': '2',
    'SUV': '3',
    'Hatchback': '4'
}

fuel_type_mapping = {
    'Diesel': '1',
    'Electric': '2',
    'Petrol': '3',
    'CNG': '4'
}

preferred_mode_mapping = {
    'Cab': '1',
    'Shuttle': '2'
}

employee_company_mapping = {
    'Cognizant': '1',
    'Infosys': '2',
    'TCS': '3',
    'Persistent': '4',
    'Wipro': '5',
    'Capgemini': '6',
    'Tech Mahindra': '7'
}

pickup_address_mapping = {
    'Baner, Pune': '1',
    'Kothrud, Pune': '2',
    'Magarpatta, Pune': '3',
    'Hinjewadi, Pune': '4',
    'Kharadi, Pune': '5',
    'Viman Nagar, Pune': '6',
    'Wakad, Pune': '7',
    'Hadapsar, Pune': '8',
    'Aundh, Pune': '9'
}

drop_company_mapping = { 
    'TCS': '3',
    'Infosys': '2',
    'Cognizant': '1',
    'Persistent': '4',
    'Wipro': '5',
    'Capgemini': '6',
    'Tech Mahindra': '7'
}

drop_address_mapping = {
    'TCS, Cybercity, Magarpatta, Pune': '1',
    'Infosys, Phase 1, Hinjewadi Rajiv Gandhi Infotech Park, Pune': '2',
    'Cognizant, EON IT Park, Kharadi, Pune': '3',
    'Persistent Systems, Senapati Bapat Road, Pune': '4',
    'Wipro, Phase 2, Hinjewadi, Pune': '5',
    'Capgemini, Talwade IT Park, Pune': '6',
    'Tech Mahindra, Rajiv Gandhi Infotech Park, Hinjewadi, Pune': '7'
}

# Apply mappings
df_selected['Shift_Type'] = df_selected['Shift_Type'].map(shift_type_mapping)
df_selected['Vehicle_Type'] = df_selected['Vehicle_Type'].map(vehicle_type_mapping)
df_selected['Fuel_Type'] = df_selected['Fuel_Type'].map(fuel_type_mapping)
df_selected['Preferred_Mode'] = df_selected['Preferred_Mode'].map(preferred_mode_mapping)
df_selected['Employee_Company'] = df_selected['Employee_Company'].map(employee_company_mapping)
df_selected['Pickup_Address'] = df_selected['Pickup_Address'].map(pickup_address_mapping)
df_selected['Drop_Company'] = df_selected['Drop_Company'].map(drop_company_mapping)
df_selected['Drop_Address'] = df_selected['Drop_Address'].map(drop_address_mapping)

# Select relevant features
selected_features = ['Employee_ID', 'Shift_Type']
target = ['Distance_km', 'Duration_min', 'Vehicle_ID', 'Vehicle_Type', 
          'Fuel_Type', 'Preferred_Mode', 'Pickup_Address', 'Drop_Company', 
          'Drop_Address', 'Pickup_Time', 'Ride_Cost', 'Price_Revenue']

# Ensure that df contains all required columns
df_selected = df_selected[selected_features + target]  # Select both features and target from df

# Define X (input) and y (output)
X_notif = df_selected[selected_features]
y_notif = df_selected[target]

# Split data into training and testing sets
X_train_notif, X_test_notif, y_train_notif, y_test_notif = train_test_split(
    X_notif, y_notif, test_size=0.2, random_state=42
)

# Shift Notification Prediction
notif_model = RandomForestRegressor(n_estimators=100, random_state=42)
notif_model.fit(X_train_notif, y_train_notif)

# Predict
y_pred_notif = notif_model.predict(X_test_notif)

# Load datasets
df_path = "df.csv"
df_encoded_path = "df_encoded.csv"

try:
    df = pd.read_csv(df_path)
    df_encoded = pd.read_csv(df_encoded_path)
except Exception as e:
    st.error(f"Error loading datasets: {e}")

# Define categorical columns
categorical_columns = [
    "Vehicle_Type", "Fuel_Type", "Pickup_Address", "Drop_Company",
    "Drop_Address", "Pickup_Time"
]

# Create decoding dictionary (encoded value -> original value)
decoded_dict = {}
for col in categorical_columns:
    if col in df.columns and col in df_encoded.columns:
        decoded_dict[col] = dict(zip(df_encoded[col], df[col]))

# Create Shift_Type encoding and decoding mappings
shift_type_mapping = {val: idx for idx, val in enumerate(df["Shift_Type"].unique())}
shift_type_reverse_mapping = {idx: val for val, idx in shift_type_mapping.items()}

# Streamlit UI
st.title("🚗 Routematic Shift Ride Notification Company Employee")

# User input
employee_id = st.selectbox("Enter the Employee ID Number", df["Employee_ID"].unique())
shift_type = st.selectbox("Select Shift Type", df["Shift_Type"].unique())

# Convert Shift_Type to numerical
shift_type_encoded = shift_type_mapping[shift_type]

# Prepare input for prediction
input_data = np.array([[employee_id, shift_type_encoded]])

# Prediction
if st.button("Predict"):
    try:
        # Make prediction
        prediction = notif_model.predict(input_data)[0]

        # Decode prediction results
        result = {

            "Pickup Time": df["Pickup_Time"].get(int(prediction[11]), "Unknown"),
            "Pickup Address": decoded_dict["Pickup_Address"].get(int(prediction[5]), "Unknown"),
            "Drop Company": decoded_dict["Drop_Company"].get(int(prediction[6]), "Unknown"),
            "Drop Address": decoded_dict["Drop_Address"].get(int(prediction[7]), "Unknown"),
              "Distance (km)": round(prediction[0], 2),
              "Duration (min)": round(prediction[1], 2)
     
        }

        # Display results in a user-friendly format
        st.success("🚀 Routematic Ride Shift Notification Company Employee")
        result_df = pd.DataFrame([result])
        st.dataframe(result_df)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
