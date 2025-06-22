import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Electricity Demand Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.8rem;
        color: #34495e;
        margin-top: 2rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3498db;
    }
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# Function to generate sample data if no data is available
@st.cache_data
def generate_sample_data():
    # Generate dates for the last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Number of cities
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    
    # Create empty dataframe
    data = []
    
    for city in cities:
        # Base demand patterns
        base_demand = 1000 + np.random.normal(0, 50, size=len(dates))
        
        # Add time patterns (daily, weekly, seasonal)
        for i, date in enumerate(dates):
            # Daily pattern (peak during day, low at night)
            hour_factor = -np.cos(date.hour * 2 * np.pi / 24) * 300
            
            # Weekly pattern (lower on weekends)
            weekday_factor = -50 if date.weekday() >= 5 else 0
            
            # Seasonal pattern (higher in summer and winter)
            month = date.month
            if month in [6, 7, 8]:  # Summer
                seasonal_factor = 200
            elif month in [12, 1, 2]:  # Winter
                seasonal_factor = 150
            else:
                seasonal_factor = 0
                
            # Temperature simulation
            if month in [6, 7, 8]:  # Summer
                temp_base = 30
            elif month in [12, 1, 2]:  # Winter
                temp_base = 5
            else:
                temp_base = 15
                
            # Add hour variation to temperature
            hour_temp_factor = -np.cos(date.hour * 2 * np.pi / 24) * 5
            temperature = temp_base + hour_temp_factor + np.random.normal(0, 2)
            
            # Calculate final demand
            demand = base_demand[i] + hour_factor + weekday_factor + seasonal_factor
            
            # Add random noise
            demand += np.random.normal(0, 30)
            
            # Ensure demand is positive
            demand = max(100, demand)
            
            # Add to data
            data.append({
                'local_time': date,
                'city': city,
                'demand': demand,
                'temperature': temperature,
                'hour': date.hour,
                'day_of_week': date.weekday(),
                'month': date.month,
                'is_weekend': 1 if date.weekday() >= 5 else 0,
                'is_holiday': 0  # Simplified
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create additional features
    df['demand_lag_24'] = df.groupby('city')['demand'].shift(24)
    df['temp_lag_24'] = df.groupby('city')['temperature'].shift(24)
    
    # Scale the data
    scaler = StandardScaler()
    numeric_cols = ['demand', 'temperature', 'demand_lag_24', 'temp_lag_24']
    df[numeric_cols] = df.groupby('city')[numeric_cols].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).reshape(-1)
    )
    
    return df

# Load your preprocessed DataFrame here
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("scaleddataset.csv", parse_dates=["local_time"])
        df.set_index("local_time", inplace=True)
        return df.reset_index()  # Return with local_time as a column
    except FileNotFoundError:
        st.warning("Dataset file not found. Using generated sample data instead.")
        return generate_sample_data()

# Create a model training function
def train_forecast_models(df_selected, features, forecast_horizon, look_back):
    """
    Train and forecast using multiple models
    Returns predictions and true values for evaluation
    """
    # Extract target and features
    y = df_selected['demand'].values
    X = df_selected[features].values
    
    # Split into train and test
    train_size = len(y) - forecast_horizon
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    results = {
        'true_values': y_test,
        'predictions': {},
        'models': {}
    }
    
    # LSTM Model
    if 'lstm' in selected_models:
        with st.spinner('Training LSTM model...'):
            # Reshape data for LSTM [samples, time steps, features]
            X_lstm = []
            y_lstm = []
            
            for i in range(look_back, len(X_train)):
                X_lstm.append(X_train[i-look_back:i])
                y_lstm.append(y_train[i])
                
            X_lstm = np.array(X_lstm)
            y_lstm = np.array(y_lstm)
            
            # Build LSTM model
            model_lstm = Sequential([
                LSTM(50, activation='relu', input_shape=(look_back, X_train.shape[1])),
                Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='mse')
            
            # Ensure we have adequate data
            if len(X_lstm) > 0:
                # Train model
                model_lstm.fit(X_lstm, y_lstm, epochs=10, batch_size=min(32, len(X_lstm)), verbose=0)
            
            # Prepare test data for prediction
            X_test_lstm = []
            for i in range(look_back, len(X_test) + look_back):
                if i < len(X):
                    X_test_lstm.append(X[i-look_back:i])
                else:
                    # For forecasting beyond available data
                    # Use previous predictions (simplified approach)
                    temp_seq = list(X[i-look_back:])
                    while len(temp_seq) < look_back:
                        temp_seq.append(temp_seq[-1])
                    X_test_lstm.append(temp_seq)
                    
            X_test_lstm = np.array(X_test_lstm)
            
            # Predict
            if len(X_test_lstm) > 0:
                y_pred_lstm = model_lstm.predict(X_test_lstm).flatten()
                results['predictions']['lstm'] = y_pred_lstm
                results['models']['lstm'] = model_lstm
    
    # ARIMA Model
    if 'arima' in selected_models:
        with st.spinner('Training ARIMA model...'):
            try:
                arima = ARIMA(y_train, order=(5,1,0))
                arima_fit = arima.fit()
                y_pred_arima = arima_fit.forecast(steps=forecast_horizon)
                results['predictions']['arima'] = y_pred_arima
                results['models']['arima'] = arima_fit
            except:
                st.warning("ARIMA model failed to converge. Skipping.")
    
    # SARIMA Model
    if 'sarima' in selected_models:
        with st.spinner('Training SARIMA model...'):
            try:
                sarima = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,24))
                sarima_fit = sarima.fit(disp=False)
                y_pred_sarima = sarima_fit.forecast(steps=forecast_horizon)
                results['predictions']['sarima'] = y_pred_sarima
                results['models']['sarima'] = sarima_fit
            except:
                st.warning("SARIMA model failed to converge. Skipping.")
    
    # Ensemble prediction (average of all models)
    if len(results['predictions']) > 0:
        ensemble_pred = np.zeros(forecast_horizon)
        for model_name, preds in results['predictions'].items():
            if len(preds) == forecast_horizon:
                ensemble_pred += preds
        
        ensemble_pred /= len(results['predictions'])
        results['predictions']['ensemble'] = ensemble_pred
    
    return results

# Calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(0.1, np.abs(y_true)))) * 100
    return mae, rmse, mape

# Main application
def main():
    df = load_data()
    
    # Title with custom styling
    st.markdown('<h1 class="main-header">⚡ Electricity Demand Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for selections
    st.sidebar.title("Settings")
    
    # City selection
    cities = sorted(df['city'].unique())
    city = st.sidebar.selectbox("Select City", cities)
    
    # Date range selection
    min_date = df['local_time'].min().date()
    max_date = df['local_time'].max().date()
    
    # Default to last 7 days in the dataset
    default_end_date = max_date
    default_start_date = max_date - timedelta(days=7)
    
    start_date = st.sidebar.date_input("Start Date", value=default_start_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=default_end_date, min_value=start_date, max_value=max_date)
    
    # Convert dates to datetime
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    
    # Filter data
    df_selected = df[(df['city'] == city) & 
                     (df['local_time'] >= start_datetime) & 
                     (df['local_time'] <= end_datetime)].copy()
    
    if len(df_selected) == 0:
        st.error(f"No data available for {city} between {start_date} and {end_date}. Please select a different date range.")
        return
    
    # Model parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Parameters")
    
    # LSTM parameters
    look_back = st.sidebar.slider("LSTM Look-back Window (hours)", 1, 72, value=24)
    
    # Clustering parameters
    k_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, value=3)
    
    # Forecast horizon
    forecast_horizon = st.sidebar.slider("Forecast Horizon (hours)", 6, 168, value=24)
    
    # Model selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Selection")
    
    # Let user select which models to use
    global selected_models
    selected_models = []
    
    if st.sidebar.checkbox("Use LSTM", value=True):
        selected_models.append('lstm')
    
    if st.sidebar.checkbox("Use ARIMA", value=True):
        selected_models.append('arima')
    
    if st.sidebar.checkbox("Use SARIMA", value=True):
        selected_models.append('sarima')
    
    # Features selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Features Selection")
    
    # Ensure essential time features exist in the dataframe
    if 'hour' not in df_selected.columns:
        df_selected['hour'] = df_selected['local_time'].dt.hour
    if 'day_of_week' not in df_selected.columns:
        df_selected['day_of_week'] = df_selected['local_time'].dt.dayofweek
    if 'month' not in df_selected.columns:
        df_selected['month'] = df_selected['local_time'].dt.month
    if 'is_weekend' not in df_selected.columns:
        df_selected['is_weekend'] = (df_selected['local_time'].dt.dayofweek >= 5).astype(int)
    
    # Create lag features if they don't exist
    if 'demand_lag_24' not in df_selected.columns:
        df_selected['demand_lag_24'] = df_selected['demand'].shift(24)
    if 'temp_lag_24' not in df_selected.columns and 'temperature' in df_selected.columns:
        df_selected['temp_lag_24'] = df_selected['temperature'].shift(24)
    
    # Fill NaN values in lag features
    for col in ['demand_lag_24', 'temp_lag_24']:
        if col in df_selected.columns:
            df_selected[col] = df_selected[col].fillna(method='bfill')
    
    # Get available features from the dataframe
    available_features = [col for col in df_selected.columns if col not in ['local_time', 'city', 'cluster', 'cluster_str']]
    
    # Default features that should work with most datasets
    default_features = ['demand']
    if 'temperature' in available_features:
        default_features.append('temperature')
    if 'hour' in available_features:
        default_features.append('hour')
    
    # Let user select features for models
    features_for_model = st.sidebar.multiselect(
        "Select Features for Modeling",
        available_features,
        default=default_features
    )
    
    # Ensure we have at least one feature
    if not features_for_model:
        features_for_model = ['demand']
        st.warning("At least one feature is required. Using 'demand' by default.")
    
    # Clustering features
    clustering_available_features = [col for col in available_features if col not in ['local_time', 'city']]
    
    features_for_clustering = st.sidebar.multiselect(
        "Select Features for Clustering",
        clustering_available_features,
        default=['demand'] if 'demand' in clustering_available_features else clustering_available_features[:1]
    )
    
    # Ensure we have at least one feature for clustering
    if not features_for_clustering:
        features_for_clustering = ['demand']
        st.warning("At least one feature is required for clustering. Using 'demand' by default.")
    
    # Main content area - use columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="subheader">📊 Historical Demand Data</h2>', unsafe_allow_html=True)
        
        # Plot historical data
        fig_history = go.Figure()
        fig_history.add_trace(go.Scatter(
            x=df_selected['local_time'],
            y=df_selected['demand'],
            mode='lines',
            name='Historical Demand',
            line=dict(color='royalblue', width=2)
        ))
        
        fig_history.update_layout(
            title=f"Electricity Demand in {city}",
            xaxis_title="Date",
            yaxis_title="Demand",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_history, use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="subheader">🌡️ Temperature vs. Demand</h2>', unsafe_allow_html=True)
        
        # Plot temperature vs demand
        fig_temp_demand = px.scatter(
            df_selected, 
            x='temperature', 
            y='demand', 
            color='hour',
            color_continuous_scale='viridis',
            title=f"Temperature vs. Demand Relationship in {city}"
        )
        
        fig_temp_demand.update_layout(
            xaxis_title="Temperature",
            yaxis_title="Demand",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_temp_demand, use_container_width=True)
    
    # Clustering section
    st.markdown('<h2 class="subheader">🔍 Demand Pattern Clustering</h2>', unsafe_allow_html=True)
    
    # Perform clustering
    try:
        with st.spinner('Performing clustering...'):
            # Ensure we have at least 2 features for clustering
            if len(features_for_clustering) < 2:
                # If only one feature selected, create a derived feature
                if 'demand' in df_selected.columns:
                    df_selected['demand_diff'] = df_selected['demand'].diff().fillna(0)
                    features_for_clustering.append('demand_diff')
                elif 'temperature' in df_selected.columns:
                    df_selected['temp_diff'] = df_selected['temperature'].diff().fillna(0)
                    features_for_clustering.append('temp_diff')
                else:
                    # Create a time-based feature
                    df_selected['hour_sin'] = np.sin(2 * np.pi * df_selected['local_time'].dt.hour / 24)
                    features_for_clustering.append('hour_sin')
                
                st.info(f"Added {features_for_clustering[-1]} as a second feature for clustering since PCA requires at least 2 features.")
            
            # Check if we have enough samples
            if len(df_selected) < 3:
                st.error("Not enough data points for clustering. Please select a wider date range.")
                return
                
            clustering_data = df_selected[features_for_clustering].dropna()
            
            # Final check if we have enough data after dropping NAs
            if clustering_data.shape[0] < 3 or clustering_data.shape[1] < 2:
                st.error(f"Insufficient data for clustering. Need at least 3 samples and 2 features, but have {clustering_data.shape[0]} samples and {clustering_data.shape[1]} features after removing missing values.")
                return
            # Standardize data for clustering
            scaler = StandardScaler()
            clustering_data_scaled = scaler.fit_transform(clustering_data)
            
            # Adjust k if we have too few samples
            actual_k = min(k_clusters, max(2, clustering_data.shape[0] // 5))
            if actual_k != k_clusters:
                st.info(f"Adjusted number of clusters to {actual_k} based on available data.")
            
            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(clustering_data_scaled)
            
            # Add cluster labels back to the dataframe
            df_selected.loc[clustering_data.index, 'cluster'] = cluster_labels
            
            # For any rows without a cluster assignment, use the nearest cluster
            if df_selected['cluster'].isna().any():
                # Fill missing clusters with the most common cluster
                most_common_cluster = df_selected['cluster'].mode()[0]
                df_selected['cluster'] = df_selected['cluster'].fillna(most_common_cluster)
            
            # Convert cluster to integer
            df_selected['cluster'] = df_selected['cluster'].astype(int)
            
            # Perform PCA for visualization
            n_components = min(2, clustering_data.shape[1])
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(clustering_data_scaled)
            
            # Create DataFrame with PCA results
            if n_components == 2:
                pca_df = pd.DataFrame(pca_result, columns=['pca1', 'pca2'])
            else:
                pca_df = pd.DataFrame(pca_result, columns=['pca1'])
                # Add a second column for visualization
                pca_df['pca2'] = np.zeros(len(pca_df))
                
            pca_df['cluster'] = kmeans.labels_
            
            # Visualize clusters
            col_cluster1, col_cluster2 = st.columns(2)
            
            with col_cluster1:
                # PCA visualization
                # Convert cluster to string for discrete coloring
                pca_df['cluster_str'] = pca_df['cluster'].astype(str)
                
                fig_pca = px.scatter(
                    pca_df, 
                    x='pca1', 
                    y='pca2', 
                    color='cluster_str',
                    title='PCA Visualization of Demand Clusters',
                    color_discrete_sequence=px.colors.qualitative.G10
                )
                
                fig_pca.update_layout(
                    xaxis_title="Principal Component 1",
                    yaxis_title="Principal Component 2",
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig_pca, use_container_width=True)
            
            with col_cluster2:
                # Time series with clusters
                df_plot = df_selected.copy()
                df_plot['datetime'] = df_plot['local_time']
                
                # Convert cluster to string for discrete coloring
                df_plot['cluster_str'] = df_plot['cluster'].astype(str)
                
                fig_cluster_ts = px.line(
                    df_plot, 
                    x='datetime', 
                    y='demand', 
                    color='cluster_str',
                    title='Demand Patterns by Cluster',
                    color_discrete_sequence=px.colors.qualitative.G10
                )
                
                fig_cluster_ts.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Demand",
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig_cluster_ts, use_container_width=True)
            
            # Cluster characteristics
            st.markdown("### Cluster Characteristics")
            
            # Calculate cluster statistics
            cluster_stats = df_selected.groupby('cluster')[features_for_clustering].agg(['mean', 'min', 'max', 'std'])
            st.dataframe(cluster_stats)
            
    except Exception as e:
        st.error(f"Error in clustering: {str(e)}")
        st.warning("Please select different features or adjust the number of clusters.")
    
    # Forecasting section
    st.markdown('<h2 class="subheader">🔮 Demand Forecasting</h2>', unsafe_allow_html=True)
    
    if not selected_models:
        st.warning("Please select at least one forecasting model from the sidebar.")
    else:
        try:
            # Train models and get forecasts
            model_results = train_forecast_models(
                df_selected, 
                features_for_model, 
                forecast_horizon, 
                look_back
            )
            
            if not model_results['predictions']:
                st.error("All models failed to train. Please try different parameters or models.")
                return
                
            # Plot forecasts
            fig_forecast = go.Figure()
            
            # Add historical data
            historical_dates = df_selected['local_time'].values
            historical_demand = df_selected['demand'].values
            
            fig_forecast.add_trace(go.Scatter(
                x=historical_dates,
                y=historical_demand,
                mode='lines',
                name='Historical Demand',
                line=dict(color='black', width=2)
            ))
            
            # Generate forecast dates
            last_date = historical_dates[-1]
            forecast_dates = [last_date + pd.Timedelta(hours=i+1) for i in range(forecast_horizon)]
            
            # Add each model's forecast
            colors = {
                'lstm': 'blue',
                'arima': 'green',
                'sarima': 'red',
                'ensemble': 'purple'
            }
            
            for model_name, predictions in model_results['predictions'].items():
                # Ensure predictions match forecast horizon
                if len(predictions) == forecast_horizon:
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=predictions,
                        mode='lines',
                        name=f'{model_name.upper()} Forecast',
                        line=dict(color=colors.get(model_name, 'orange'), width=2, dash='dash')
                    ))
            
            fig_forecast.update_layout(
                title="Electricity Demand Forecast",
                xaxis_title="Date",
                yaxis_title="Demand",
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Evaluation metrics
            st.markdown("### Forecast Evaluation Metrics")
            
            # Create metrics for each model
            metric_cols = st.columns(len(model_results['predictions']))
            
            for i, (model_name, predictions) in enumerate(model_results['predictions'].items()):
                if len(predictions) == forecast_horizon:
                    # Use all available true values for comparison
                    y_true = model_results['true_values'][:len(predictions)]
                    mae, rmse, mape = calculate_metrics(y_true, predictions)
                    
                    with metric_cols[i]:
                        st.markdown(f"**{model_name.upper()}**")
                        st.markdown(f"**MAE:** {mae:.2f}")
                        st.markdown(f"**RMSE:** {rmse:.2f}")
                        st.markdown(f"**MAPE:** {mape:.2f}%")
            
        except Exception as e:
            st.error(f"Error in forecasting: {str(e)}")
            st.warning("Please select different features or adjust the model parameters.")
    
    # Help and Documentation
    st.markdown('<h2 class="subheader">📚 Help & Documentation</h2>', unsafe_allow_html=True)
    
    with st.expander("📖 How to Use This Dashboard"):
        st.markdown("""
        **Steps to Get Started**:
        1. **Select a City**: Choose the city you want to analyze from the dropdown menu.
        2. **Choose a Date Range**: Select the start and end dates for your analysis.
        3. **Adjust Model Parameters**: 
           - LSTM Look-back Window: Number of past time steps to consider.
           - Number of Clusters (k): How many patterns to identify in your data.
           - Forecast Horizon: How far ahead to predict.
        4. **Select Models**: Choose which forecasting models to use.
        5. **Select Features**: Pick which data attributes to use for modeling and clustering.
        
        **Understanding the Visualizations**:
        - **Historical Demand**: Shows the actual electricity demand over time.
        - **Temperature vs. Demand**: Reveals the relationship between temperature and demand.
        - **Demand Pattern Clustering**: Groups similar demand patterns together.
        - **Demand Forecasting**: Predicts future demand using selected models.
        
        **Tips for Better Results**:
        - Use a longer history for more stable forecasts.
        - Include temperature in your features for more accurate predictions.
        - Try different combinations of models and features.
        - The ensemble approach often gives the most balanced forecasts.
        """)
    
    with st.expander("📊 Technical Details & Methodology"):
        st.markdown("""
        **Data Sources**:
        The application uses hourly electricity demand data with weather and time features.
        
        **Feature Engineering**:
        - **Temporal Features**: Hour of day, day of week, month, weekend indicator.
        - **Weather Features**: Temperature.
        - **Lagged Features**: Previous day's demand and temperature.
        
        **Clustering Methodology**:
        - **K-Means**: Partitioning observations into k clusters to minimize within-cluster variance.
        - **PCA**: Dimensionality reduction for visualizing high-dimensional data in 2D.
        
        **Forecasting Models**:
        - **LSTM** (Long Short-Term Memory): A recurrent neural network architecture designed to model temporal sequences and long-range dependencies.
          - Architecture: 50 LSTM units followed by a dense output layer.
          - Training: Adam optimizer, MSE loss function, 10 epochs.
        
        - **ARIMA** (AutoRegressive Integrated Moving Average):
          - Default order: (5,1,0) - AR(5), I(1), MA(0).
          - Works well for non-seasonal time series with trends.
        
        - **SARIMA** (Seasonal ARIMA):
          - Default order: (1,1,1)x(1,1,1,24) - includes seasonal components.
          - Captures both trend and seasonality (daily patterns).
        
        - **Ensemble**: Averages predictions from all selected models to reduce individual model biases.
        
        **Evaluation Metrics**:
        - **MAE** (Mean Absolute Error): Average magnitude of errors without considering direction.
        - **RMSE** (Root Mean Squared Error): Square root of the average of squared differences.
        - **MAPE** (Mean Absolute Percentage Error): Percentage representation of error.
        
        **Libraries Used**:
        - Data Handling: Pandas, NumPy
        - Visualization: Plotly, Matplotlib, Seaborn
        - Machine Learning: Scikit-learn, TensorFlow/Keras
        - Time Series: Statsmodels
        - Web Application: Streamlit
        """)

if __name__ == "__main__":
    main()