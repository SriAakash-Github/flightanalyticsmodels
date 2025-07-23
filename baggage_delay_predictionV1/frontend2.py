import streamlit as st
import requests
from datetime import datetime
import pandas as pd
import os
import io
import plotly.graph_objects as go
import plotly.express as px

# --- App config ---
st.set_page_config(page_title='Baggage Delay Prediction', page_icon='🧳', layout='wide')

# --- Local storage for history ---
HISTORY_FILE = 'prediction_history.csv'

# --- Helper for time validation ---
def is_valid_time(t):
    try:
        datetime.strptime(t.strip(), '%H:%M')
        return True
    except ValueError:
        return False

# --- Sidebar dashboard switch ---
dashboard_options = ['Baggage Delay', 'Airline Delay']
dashboard_selected = st.sidebar.selectbox('Select Dashboard', dashboard_options)

# --- Sidebar navigation ---
menu = {
    'Summary': '🏠',
    'Details': '🧳',
    'History': '📊',
    'Settings': '⚙️'
}
st.sidebar.markdown('<style>.sidebar-title{font-size:1.3rem;font-weight:700;margin-bottom:1.5rem;}</style>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
selected = st.sidebar.radio(
    '',
    list(menu.keys()),
    format_func=lambda x: f"{menu[x]}  {x}"
)

# --- Custom CSS for styling ---
st.markdown('''
    <style>
    .main-header {
        background: #2563eb;
        color: white;
        padding: 1.2rem 2rem 1.2rem 2rem;
        border-radius: 0 0 16px 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        position: relative;
    }
    .main-header h1 {
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .main-header .subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.2rem;
        color: #e0e7ff;
    }
    .details-card {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        padding: 2rem 2.5rem;
        margin-top: 1.5rem;
    }
    .details-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2563eb;
        margin-bottom: 1.2rem;
    }
    .stButton>button {
        background: #2563eb;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        border: none;
        margin-top: 1rem;
    }
    </style>
''', unsafe_allow_html=True)

# --- Main header ---
if dashboard_selected == 'Baggage Delay':
    st.markdown('''
        <div class="main-header">
            <h1>🧳 Baggage Delay Prediction</h1>
            <div class="subtitle">Predict if your baggage will be delayed and estimate the delay time based on your flight and check-in details.</div>
        </div>
    ''', unsafe_allow_html=True)
else:
    st.markdown('''
        <div class="main-header">
            <h1>✈️ Airline Delay Prediction</h1>
            <div class="subtitle">Predict if your flight will be delayed. (Coming soon!)</div>
        </div>
    ''', unsafe_allow_html=True)

# --- Dashboard content switch ---
if dashboard_selected == 'Baggage Delay':
    # --- Summary Tab ---
    if selected == 'Summary':
        st.markdown('<div class="details-card">', unsafe_allow_html=True)
        st.markdown('<div class="details-title">Welcome to the Baggage Delay Prediction Dashboard</div>', unsafe_allow_html=True)
        st.write("Use the sidebar to navigate between prediction, history, and settings.")
        # Show some stats if history exists
        if os.path.exists(HISTORY_FILE):
            df_hist = pd.read_csv(HISTORY_FILE)
            st.subheader('📈 Recent Prediction Stats')
            st.metric('Total Predictions', len(df_hist))
            st.metric('Delayed Predictions', int(df_hist['prediction'].sum()))
            st.metric('Average Delay (min)', f"{df_hist['predicted_delay_minutes'].mean():.1f}")
            # --- Analytics/Charts ---
            st.markdown('---')
            st.subheader('Analytics')
            # Pie chart: Delayed vs Not Delayed
            st.markdown('**Delayed vs Not Delayed (Pie Chart)**')
            pie_df = df_hist['prediction'].value_counts().rename({0:'Not Delayed',1:'Delayed'}).reset_index()
            pie_df.columns = ['Status','Count']
            fig = go.Figure(data=[go.Pie(labels=pie_df['Status'], values=pie_df['Count'])])
            fig.update_layout(title="Delayed vs Not Delayed")
            st.plotly_chart(fig)
            # Bar chart: Avg delay by origin airport
            if 'origin_airport' in df_hist.columns:
                st.markdown('**Average Delay by Origin Airport (Bar Chart)**')
                bar_df = df_hist.groupby('origin_airport')['predicted_delay_minutes'].mean().reset_index()
                st.bar_chart(bar_df.rename(columns={'origin_airport':'index'}).set_index('index'))
            # 2. Delay rate by weather condition
            if 'weather_condition' in df_hist.columns:
                st.markdown('**Delay Rate by Weather Condition (Bar Chart)**')
                delay_rate_weather = df_hist.groupby('weather_condition')['prediction'].mean().reset_index()
                delay_rate_weather['delay_rate'] = delay_rate_weather['prediction'] * 100
                st.bar_chart(delay_rate_weather.rename(columns={'weather_condition':'index'}).set_index('index')['delay_rate'])
            # 4. Distribution of predicted delay minutes
            if 'predicted_delay_minutes' in df_hist.columns:
                st.markdown('**Distribution of Predicted Delay Minutes (Histogram)**')
                fig_hist = px.histogram(df_hist, x='predicted_delay_minutes', nbins=30, title='Distribution of Predicted Delay Minutes')
                st.plotly_chart(fig_hist)
            # 5. Recent prediction trends (number of predictions per week)
            if 'timestamp' in df_hist.columns:
                st.markdown('**Recent Prediction Trends (Line Chart)**')
                df_hist['date'] = pd.to_datetime(df_hist['timestamp']).dt.date
                trend_df = df_hist.groupby('date').size().reset_index(name='num_predictions')
                st.line_chart(trend_df.rename(columns={'date':'index'}).set_index('index')['num_predictions'])
        else:
            st.info('No predictions made yet.')
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Details Tab (Prediction Form & Batch) ---
    if selected == 'Details':
        st.markdown('<div class="details-card">', unsafe_allow_html=True)
        st.markdown('<div class="details-title">Passenger Details</div>', unsafe_allow_html=True)
        pd_col1, pd_col2 = st.columns(2)
        with pd_col1:
            passenger_name = st.text_input('Passenger Name')
            flight_number = st.text_input('Flight Number')
        with pd_col2:
            baggage_tag_number = st.text_input('Baggage Tag Number')
            special_handling = st.text_input('Special Handling Instructions')
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="details-card">', unsafe_allow_html=True)
        st.markdown('<div class="details-title">Flight & Baggage Details</div>', unsafe_allow_html=True)
        airport_options = ['DEL', 'FRA', 'JFK', 'ORD', 'SFO', 'DXB', 'BLR', 'LHR']
        weather_options = ['Clear', 'Rain', 'Snow', 'Storm', 'Fog']
        extra_baggage_options = ['Yes', 'No']
        day_of_week_options = [
            ('0', 'Monday'), ('1', 'Tuesday'), ('2', 'Wednesday'),
            ('3', 'Thursday'), ('4', 'Friday'), ('5', 'Saturday'), ('6', 'Sunday')
        ]
        col1, col2 = st.columns(2)
        with col1:
            origin_airport = st.selectbox('Origin Airport', airport_options)
            destination_airport = st.selectbox('Destination Airport', airport_options)
            departure_time = st.text_input('Scheduled Departure Time (HH:MM)', value='12:00')
            actual_departure_time = st.text_input('Actual Departure Time (HH:MM)', value='12:00')
            arrival_time = st.text_input('Scheduled Arrival Time (HH:MM)', value='15:00')
            check_in_time = st.text_input('Check-in Time (HH:MM)', value='10:00')
            travel_duration_min = st.number_input('Travel Duration (min)', min_value=1, value=60)
        with col2:
            weather_condition = st.selectbox('Weather Condition', weather_options)
            number_of_bags = st.number_input('Number of Bags', min_value=1, max_value=10, value=1)
            bag_weight_kg = st.number_input('Bag Weight (kg)', min_value=0.0, value=10.0)
            day_of_week = st.selectbox('Day of Week', [d[0] for d in day_of_week_options], format_func=lambda x: dict(day_of_week_options)[x])
            is_international = st.selectbox('International Flight?', ['Yes', 'No'])
            extra_baggage = st.selectbox('Extra Baggage', extra_baggage_options)
        # --- Form Validation ---
        errors = []
        if origin_airport not in airport_options:
            errors.append('Invalid origin airport.')
        if destination_airport not in airport_options:
            errors.append('Invalid destination airport.')
        if not all(is_valid_time(t) for t in [departure_time, actual_departure_time, arrival_time, check_in_time]):
            errors.append('All time fields must be in HH:MM format.')
        if not (1 <= number_of_bags <= 10):
            errors.append('Number of bags must be between 1 and 10.')
        if bag_weight_kg < 0.0:
            errors.append('Bag weight must be non-negative.')
        if travel_duration_min <= 0:
            errors.append('Travel duration must be a positive number.')
        # --- Predict Button ---
        predict_clicked = st.button('Predict Baggage Delay', use_container_width=True)
        # --- Results Section ---
        if predict_clicked:
            if errors:
                for err in errors:
                    st.error(err)
            else:
                data = {
                    'origin_airport': origin_airport,
                    'destination_airport': destination_airport,
                    'departure_time': departure_time.strip(),
                    'actual_departure_time': actual_departure_time.strip(),
                    'arrival_time': arrival_time.strip(),
                    'check_in_time': check_in_time.strip(),
                    'weather_condition': weather_condition,
                    'number_of_bags': number_of_bags,
                    'bag_weight_kg': bag_weight_kg,
                    'day_of_week': day_of_week,
                    'is_international': 1 if is_international == 'Yes' else 0,
                    'extra_baggage': extra_baggage,
                    'travel_duration_min': travel_duration_min
                }
                try:
                    with st.spinner('Predicting...'):
                        response = requests.post('http://localhost:5000/predict', json=data)
                        result = response.json()
                        reg_response = requests.post('http://localhost:5000/predict_reg', json=data)
                        reg_result = reg_response.json()
                        delay_minutes = reg_result.get('predicted_delay_minutes', None)
                    st.markdown('<hr>', unsafe_allow_html=True)
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        if result['prediction'] == 1:
                            st.error(f"❗ **Prediction: Baggage Delayed**\n\n**Probability:** {result['probability']:.2%}")
                        else:
                            st.success(f"✅ **Prediction: Baggage Not Delayed**\n\n**Probability:** {1-result['probability']:.2%}")
                    with res_col2:
                        if delay_minutes is not None:
                            st.info(f"🕒 **Estimated Delay:** {delay_minutes:.1f} minutes")
                    # --- Save to history ---
                    record = data.copy()
                    record['passenger_name'] = passenger_name
                    record['baggage_tag_number'] = baggage_tag_number
                    record['flight_number'] = flight_number
                    record['special_handling'] = special_handling
                    record['prediction'] = result['prediction']
                    record['probability'] = result['probability']
                    record['predicted_delay_minutes'] = delay_minutes
                    record['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    df_new = pd.DataFrame([record])
                    if os.path.exists(HISTORY_FILE):
                        df_hist = pd.read_csv(HISTORY_FILE)
                        df_hist = pd.concat([df_hist, df_new], ignore_index=True)
                    else:
                        df_hist = df_new
                    df_hist.to_csv(HISTORY_FILE, index=False)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info('Fill in your flight and baggage details, then click **Predict Baggage Delay**.')
        st.markdown('---')
        # --- Batch Prediction ---
        st.markdown('<div class="details-title">Batch Prediction (Upload CSV)</div>', unsafe_allow_html=True)
        st.write("Upload a CSV file with the same columns as the form above. You'll get predictions for all rows.")
        batch_file = st.file_uploader('Upload CSV for Batch Prediction', type=['csv'])
        if batch_file is not None:
            try:
                df_batch = pd.read_csv(batch_file)
                # Validate columns
                required_cols = ['origin_airport', 'destination_airport', 'departure_time', 'actual_departure_time', 'arrival_time', 'check_in_time', 'weather_condition', 'number_of_bags', 'bag_weight_kg', 'day_of_week', 'is_international', 'extra_baggage', 'travel_duration_min']
                if not all(col in df_batch.columns for col in required_cols):
                    st.error('CSV missing required columns.')
                else:
                    results = []
                    for idx, row in df_batch.iterrows():
                        # Validate each row
                        row_errors = []
                        if row['origin_airport'] not in airport_options:
                            row_errors.append('Invalid origin airport.')
                        if row['destination_airport'] not in airport_options:
                            row_errors.append('Invalid destination airport.')
                        for tcol in ['departure_time', 'actual_departure_time', 'arrival_time', 'check_in_time']:
                            if not is_valid_time(str(row[tcol])):
                                row_errors.append(f'Invalid time: {tcol}')
                        if not (1 <= row['number_of_bags'] <= 10):
                            row_errors.append('Number of bags out of range.')
                        if row['bag_weight_kg'] < 0.0:
                            row_errors.append('Bag weight must be non-negative.')
                        if 'travel_duration_min' in row and row['travel_duration_min'] <= 0:
                            row_errors.append('Travel duration must be positive.')
                        if row_errors:
                            results.append({**row, 'prediction': None, 'probability': None, 'predicted_delay_minutes': None, 'error': '; '.join(row_errors)})
                            continue
                        data = {k: row[k] for k in required_cols}
                        try:
                            response = requests.post('http://localhost:5000/predict', json=data)
                            result = response.json()
                            reg_response = requests.post('http://localhost:5000/predict_reg', json=data)
                            reg_result = reg_response.json()
                            delay_minutes = reg_result.get('predicted_delay_minutes', None)
                            # Add passenger details if present in batch
                            passenger_name = row.get('passenger_name', '')
                            baggage_tag_number = row.get('baggage_tag_number', '')
                            flight_number = row.get('flight_number', '')
                            special_handling = row.get('special_handling', '')
                            results.append({**row, 'prediction': result['prediction'], 'probability': result['probability'], 'predicted_delay_minutes': delay_minutes, 'passenger_name': passenger_name, 'baggage_tag_number': baggage_tag_number, 'flight_number': flight_number, 'special_handling': special_handling, 'error': ''})
                        except Exception as e:
                            results.append({**row, 'prediction': None, 'probability': None, 'predicted_delay_minutes': None, 'error': str(e)})
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)
                    st.download_button('Download Batch Results as CSV', df_results.to_csv(index=False), file_name='batch_prediction_results.csv', mime='text/csv')
                    # Save successful predictions to history
                    if os.path.exists(HISTORY_FILE):
                        df_hist = pd.read_csv(HISTORY_FILE)
                    else:
                        df_hist = pd.DataFrame()
                    df_to_save = df_results[(df_results['error'] == '')].copy()
                    df_to_save['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    if not df_to_save.empty:
                        df_hist = pd.concat([df_hist, df_to_save], ignore_index=True)
                        df_hist.to_csv(HISTORY_FILE, index=False)
            except Exception as e:
                st.error(f'Error processing batch: {e}')
        st.markdown('</div>', unsafe_allow_html=True)

    # --- History Tab ---
    if selected == 'History':
        st.markdown('<div class="details-card">', unsafe_allow_html=True)
        st.markdown('<div class="details-title">Prediction History</div>', unsafe_allow_html=True)
        if os.path.exists(HISTORY_FILE):
            df_hist = pd.read_csv(HISTORY_FILE)
            st.dataframe(df_hist, use_container_width=True)
            st.download_button('Download History as CSV', df_hist.to_csv(index=False), file_name='prediction_history.csv', mime='text/csv')
        else:
            st.info('No prediction history found.')
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Settings Tab ---
    if selected == 'Settings':
        st.markdown('<div class="details-card">', unsafe_allow_html=True)
        st.markdown('<div class="details-title">Settings & App Info</div>', unsafe_allow_html=True)
        st.write('You can clear your prediction history or view app information below.')
        if st.button('Clear Prediction History', type='primary'):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
                st.success('Prediction history cleared!')
            else:
                st.info('No history to clear.')
        st.markdown('---')
        st.write('**App Version:** 1.0')
        st.write('**Developer:** Your Name')
        st.write('**Powered by Streamlit & Flask**')
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # --- Airline Delay Dashboard Placeholder ---
    if selected == 'Summary':
        st.markdown('<div class="details-card">', unsafe_allow_html=True)
        st.markdown('<div class="details-title">Welcome to the Airline Delay Prediction Dashboard</div>', unsafe_allow_html=True)
        st.info('This dashboard will allow you to predict airline delays. (Feature coming soon!)')
        st.markdown('</div>', unsafe_allow_html=True)
    elif selected == 'Details':
        st.markdown('<div class="details-card">', unsafe_allow_html=True)
        st.markdown('<div class="details-title">Airline Delay Prediction Form</div>', unsafe_allow_html=True)
        st.info('Airline delay prediction form will be available here soon.')
        st.markdown('</div>', unsafe_allow_html=True)
    elif selected == 'History':
        st.markdown('<div class="details-card">', unsafe_allow_html=True)
        st.markdown('<div class="details-title">Airline Delay Prediction History</div>', unsafe_allow_html=True)
        st.info('History of airline delay predictions will be shown here.')
        st.markdown('</div>', unsafe_allow_html=True)
    elif selected == 'Settings':
        st.markdown('<div class="details-card">', unsafe_allow_html=True)
        st.markdown('<div class="details-title">Settings & App Info</div>', unsafe_allow_html=True)
        st.info('Settings for airline delay prediction will be available here.')
        st.markdown('</div>', unsafe_allow_html=True)
