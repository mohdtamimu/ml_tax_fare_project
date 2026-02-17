import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="🚕 Taxi Fare Prediction",
    page_icon="🚖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===== SIDEBAR =====
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/744/744922.png", width=150)
st.sidebar.title("Taxi Fare Prediction")
st.sidebar.markdown("""
Predict your taxi fare in real-time.
Fill in the trip details and click **Predict**.
""")

# ===== MAIN HEADER =====
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🚕 Taxi Fare Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Estimate your taxi fare instantly based on your trip details.</p>", unsafe_allow_html=True)
st.markdown("---")

# ===== USER INPUT FORM =====
with st.form(key='fare_form'):
    st.subheader("Trip Details")
    
    col1, col2 = st.columns(2)
    with col1:
        pickup_longitude = st.number_input("Pickup Longitude", value=-73.985, format="%.6f")
        pickup_latitude = st.number_input("Pickup Latitude", value=40.758, format="%.6f")
        dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.981, format="%.6f")
        dropoff_latitude = st.number_input("Dropoff Latitude", value=40.764, format="%.6f")
    
    with col2:
        passenger_count = st.slider("Passenger Count", min_value=1, max_value=6, value=1)
        year = st.slider("Year", min_value=2009, max_value=2026, value=2026)
        day = st.slider("Day of Month", min_value=1, max_value=31, value=15)
        dayOfWeek = st.slider("Day of Week (Mon=1, Sun=7)", min_value=1, max_value=7, value=2)
        hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=14)
    
    day_name = st.selectbox("Day Name", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    time_of_day = st.selectbox("Time of Day", ["Morning","Midday","Evening","Night"])
    
    submit_button = st.form_submit_button(label="Predict Fare 🚖")

# ===== ENCODE CATEGORICAL =====
day_name_dict = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
time_of_day_dict = {"Morning":0,"Midday":1,"Evening":2,"Night":3}

day_name_enc = day_name_dict[day_name]
time_of_day_enc = time_of_day_dict[time_of_day]

# ===== FEATURE ENGINEERING =====
distance = np.sqrt((dropoff_latitude - pickup_latitude)**2 + (dropoff_longitude - pickup_longitude)**2)

# ===== PREDICTION =====
if submit_button:
    features = np.array([[pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
                          passenger_count, year, day, dayOfWeek, hour, day_name_enc, time_of_day_enc, distance]])
    
    prediction = model.predict(features)
    
    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center; color: green;'>Estimated Fare: <b>${prediction[0]:.2f}</b></h2>", unsafe_allow_html=True)
