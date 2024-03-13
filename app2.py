import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import joblib




import sys
st.write(sys.executable)

random_forest_model = load('models/regressors/Random Forest Regressor.joblib')
label_encoders = {col: load(f'models/encoders/label_encoder_{col}.joblib') for col in ['Gender', 'Engine', 'Company', 'Model', 'Transmission', 'Color', 'Body Style', 'Dealer_Region']}
pca_model = load('models/pca/pca_model.joblib')

def preprocess_data(new_data, label_encoders, pca_model):
    for col in new_data.columns:
        if col in label_encoders:
            new_data[col] = label_encoders[col].transform(new_data[col])
    X_new_pca = pca_model.transform(new_data)
    return X_new_pca

@st.cache
def load_data():
    df = pd.read_csv('data/data.csv')
    return df

df = load_data()

st.title('Car Sales Analysis and Prediction App')

# Data Exploration Section
st.header('Data Exploration')
if st.checkbox('Show raw data'):
    st.write(df)

# Price Distribution by Gender Section
st.header("Price Distribution by Gender")

plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Price ($)', data=df)
plt.title('Price Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Price ($)')

st.pyplot(plt)


# Histogram for Annual Income
st.subheader('Annual Income Distribution')
fig, ax = plt.subplots()
sns.histplot(df['Annual Income'], kde=True, ax=ax)
st.pyplot(fig)

# Boxplot for Price
st.subheader('Car Price Distribution')
fig, ax = plt.subplots()
sns.boxplot(x=df['Price ($)'], ax=ax)
st.pyplot(fig)

# Car Sales by Company Section
st.header("Car Sales by Company")

plt.figure(figsize=(10, 6))
sns.countplot(y=df['Company'], order=df['Company'].value_counts().index)
plt.title('Car Sales by Company')
plt.xlabel('Count')
plt.ylabel('Company')

st.pyplot(plt)

#top brands for each gender
top_brands_men = df[df['Gender'] == 'Male']['Company'].value_counts().nlargest(3)
top_brands_women = df[df['Gender'] == 'Female']['Company'].value_counts().nlargest(3)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(x=top_brands_men.index, y=top_brands_men.values, palette='Blues_d', ax=axes[0])
axes[0].set_title('Top 3 Most Bought Car Brands by Men')
axes[0].set_xlabel('Brand')
axes[0].set_ylabel('Number of Purchases')

sns.barplot(x=top_brands_women.index, y=top_brands_women.values, palette='Reds_d', ax=axes[1])
axes[1].set_title('Top 3 Most Bought Car Brands by Women')
axes[1].set_xlabel('Brand')
axes[1].set_ylabel('Number of Purchases')

st.pyplot(fig)

# Correlation Matrix Section
st.header("Correlation Matrix")

numeric_cols = ['Price ($)', 'Annual Income']  
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

st.pyplot(plt)


def get_unique_values_for_all_columns(df, categorical_columns):
    unique_values = {}
    for column in categorical_columns:
        unique_values[column] = df[column].dropna().unique().tolist()
    return unique_values

df = pd.read_csv('data/data.csv')

categorical_columns = ['Gender', 'Engine', 'Company', 'Model', 'Transmission', 'Color', 'Body Style', 'Dealer_Region']
unique_values = get_unique_values_for_all_columns(df, categorical_columns)


st.header("Car Price Prediction")

input_data = {}
input_data['Gender'] = st.selectbox('Gender', unique_values['Gender'])
input_data['Engine'] = st.selectbox('Engine', unique_values['Engine'])
input_data['Company'] = st.selectbox('Company', unique_values['Company'])
input_data['Model'] = st.selectbox('Model', unique_values['Model'])
input_data['Transmission'] = st.selectbox('Transmission', unique_values['Transmission'])
input_data['Color'] = st.selectbox('Color', unique_values['Color'])
input_data['Body Style'] = st.selectbox('Body Style', unique_values['Body Style'])
input_data['Dealer_Region'] = st.selectbox('Dealer Region', unique_values['Dealer_Region'])


submit = st.button('Predict Price')
if submit:
    new_data = pd.DataFrame([input_data])
    X_new_pca = preprocess_data(new_data, label_encoders, pca_model)
    predicted_price = random_forest_model.predict(X_new_pca)
    st.write(f"Predicted Sale Price: ${predicted_price[0]:.2f}")

# Sales Forecast Section
st.header("Sales Forecast")
df = pd.read_csv('data/data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df_monthly = df[['Date', 'Price ($)']].resample('M', on='Date').mean().reset_index()
df_prophet = df_monthly.rename(columns={'Date': 'ds', 'Price ($)': 'y'})

model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.fit(df_prophet)

future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

fig_components = plot_components_plotly(model, forecast)
st.plotly_chart(fig_components, use_container_width=True)


# 3d plot 
st.header("3D Clustering Visualization")
@st.cache
def load_df_reduced():
    # Load the DataFrame
    return joblib.load('models/pca/df_reduced.joblib')

st.title('Car Sales Analysis and Prediction App')

# Load the reduced DataFrame
df_reduced = load_df_reduced()

# 3D Plot
st.header("3D Clustering Visualization")
fig = px.scatter_3d(df_reduced, x='PC1', y='PC2', z='PC3', color='Cluster', opacity=0.7, size_max=5, title='3D Clustering')
st.plotly_chart(fig, use_container_width=True)