import streamlit as st
import pandas as pd
import joblib


model = joblib.load("trained_model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Flight Ticket Price Prediction", page_icon="✈️")

st.title("✈️ Flight Ticket Price Prediction")
st.write("Enter the flight details below to estimate the ticket price.")



airline = st.selectbox(
    "Airline",
    ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
)

flight = st.text_input("Flight Number (e.g. 6E-148)")

source_city = st.selectbox(
    "Source City",
    ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
)

departure_time = st.selectbox(
    "Departure Time",
    ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
)

stops = st.selectbox(
    "Stops",
    [0, 1, 2]
)

arrival_time = st.selectbox(
    "Arrival Time",
    ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
)

destination_city = st.selectbox(
    "Destination City",
    ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
)

travel_class = st.selectbox(
    "Class",
    ["Economy", "Business"]
)

duration = st.number_input(
    "Duration (hours)",
    min_value=0.0,
    value=2.0,
    step=0.1,
    format="%.1f"
)

days_left = st.number_input(
    "Days Left",
    min_value=1,
    max_value=365,
    step=1
)



if st.button("Predict Ticket Price"):

   
    input_df = pd.DataFrame(0.0, index=[0], columns=columns)

  
    input_df.loc[0, "stops"] = stops
    input_df.loc[0, "duration"] = float(duration)
    input_df.loc[0, "days_left"] = days_left

  
    features = [
        f"airline_{airline}",
        f"flight_{flight}",
        f"source_city_{source_city}",
        f"departure_time_{departure_time}",
        f"arrival_time_{arrival_time}",
        f"destination_city_{destination_city}",
        f"class_{travel_class}",
    ]

    for feature in features:
        if feature in input_df.columns:
            input_df.loc[0, feature] = 1

    prediction = model.predict(input_df)[0]

    st.success(f"Estimated Ticket Price: ₹ {prediction:,.2f}")
