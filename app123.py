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
    #price distribution markdown
st.markdown("""
### Price Distribution by Gender
Here we analyze how car prices vary between different genders. This can provide insights into purchasing preferences and trends.
""")

plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Price ($)', data=df)
plt.title('Price Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Price ($)')

st.pyplot(plt)


# Histogram for Annual Income
st.header('Annual Income Distribution')
#histogram for annual income markdown
st.markdown("""
### Histogram for Annual Income

The distribution of annual income, as evidenced by the histogram, exhibits a pronounced right skew, with a majority of observations clustered at the lower end of the income spectrum. This concentration at the lower end indicates a larger segment of the customer base falls within the lower-income category. The implication for car sales strategies is that there may be greater market potential for more affordable vehicle options. 

Moreover, the presence of a long tail towards the higher income values suggests that while customers with higher incomes are fewer in number, they do represent a non-negligible market segment. This secondary market may present opportunities for higher-margin sales through luxury or higher-end vehicle models.

Dealerships positioned to offer a varied product range with flexible financing options could leverage the prevailing income distribution to maximize market penetration and profitability.
""")
fig, ax = plt.subplots()
sns.histplot(df['Annual Income'], kde=True, ax=ax)
st.pyplot(fig)



# Boxplot for Price
st.header('Car Price Distribution')
    #boxplot for car price markdown
st.markdown("""
### Car Sales by Company
The box plot for car prices substantiates the earlier observations derived from the income histogram, affirming that the majority of car transactions occur at the more accessible end of the price spectrum. The median price, slightly above $20,000, reinforces the popularity of more affordably priced vehicles. This price point likely resonates with the financial realities of the larger customer demographic, whose income levels we previously noted to be skewed towards the lower end.

Simultaneously, the identification of outliers on the higher end of the scale indicates the existence of a niche yet viable market for premium vehicles. These outliers may correspond to luxury vehicles or models with advanced features that cater to customers with more substantial incomes.

Strategically, the data portrayed in the box plot suggests that the car dealership market is predominantly composed of sales from lower to mid-range vehicles. However, it's imperative not to overlook the opportunity presented by the high-priced outliers. A balanced inventory that caters to the primary market while also offering select high-end models could capitalize on the full breadth of the market's potential.

""")

fig, ax = plt.subplots()
sns.boxplot(x=df['Price ($)'], ax=ax)
st.pyplot(fig)



# Car Sales by Company Section
st.header("Car Sales by Company")
st.markdown("""           
This horizontal bar chart offers a clear visual representation of car sales across different automotive brands within our dataset, providing valuable insights for clients interested in establishing a new car dealership.

Key takeaways from the chart include:

Dominant Players: The market leaders—Chevrolet, Dodge, and Ford—stand out with the highest number of sales, indicative of robust consumer demand for these brands. This trend suggests that aligning with these brands could be a promising starting point for a new dealership, given their widespread popularity.

Niche Opportunities: Brands with fewer sales may represent specialized market segments or cater to a clientele with unique preferences, potentially commanding higher margins. These niche segments offer opportunities for a dealership to differentiate itself in the marketplace.

Strategic Inventory Management: The chart serves as a strategic guide for inventory selection, signaling which brands' vehicles are likely to be the most popular. A well-curated mix of top-selling and niche vehicles could maximize sales potential while minimizing inventory risks.
 """)

plt.figure(figsize=(10, 6))
sns.countplot(y=df['Company'], order=df['Company'].value_counts().index)
plt.title('Car Sales by Company')
plt.xlabel('Count')
plt.ylabel('Company')

st.pyplot(plt)


#top brands for each gender
st.markdown("""
### Top 3 car brands by gender

These 2 models support the bar chart showing that both women and men has the same brands in there top 3 with different rankings. 
The dataset consists of fewer women than men suggesting that there may be a significant difference in whom buys cars the most. 

""")

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
st.markdown("""
### Correlation matrix	
The correlation matrix visualizes the correlation between price and annual income. 
The matrix suggest that there is almost no correlation between these suggesting that we may potentially have costumers with higher income buying lower priced cars.

""")

numeric_cols = ['Price ($)', 'Annual Income']  
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

st.pyplot(plt)


#region slecetion starts here
st.header("Regional Sales Analysis")
# Car Sales by Region
st.header('Car Sales by Region')
def plot_sales_by_region(df):
    # Calculate and plot sales by region
    sales_by_region = df.groupby('Dealer_Region')['Price ($)'].sum()
    fig, ax = plt.subplots()
    sns.barplot(x=sales_by_region.index, y=sales_by_region.values, ax=ax)
    ax.set_title('Total Sales by Region')
    ax.set_xlabel('Region')
    ax.set_ylabel('Sales')
    return fig

st.pyplot(plot_sales_by_region(df))

# Median Price Data for Austin
st.header('Median Price Data for Austin')
def plot_austin_data(df):
    austin_dealerships = df[df['Dealer_Region'] == 'Austin']
    austin_data = austin_dealerships.groupby('Dealer_Name')['Price ($)'].sum()
    median_price = austin_data.median()
    fig, ax = plt.subplots()
    austin_data.plot(kind='bar', ax=ax)
    ax.axhline(median_price, color='red', linestyle='--', label='Median Price')
    ax.set_title('Austin Data with Median')
    ax.legend()
    return fig

st.pyplot(plot_austin_data(df))

# Scottsdale Data with Median
st.header('Median Price Data for Scottsdale')
def plot_scottsdale_data(df):
    scottsdale_dealerships = df[df['Dealer_Region'] == 'Scottsdale']
    scottsdale_data = scottsdale_dealerships.groupby('Dealer_Name')['Price ($)'].sum()
    median_price = scottsdale_data.median()
    fig, ax = plt.subplots()
    scottsdale_data.plot(kind='bar', ax=ax)
    ax.axhline(median_price, color='red', linestyle='--', label='Median Price')
    ax.set_title('Scottsdale Data with Median')
    ax.set_xlabel('Dealer Name')
    ax.set_ylabel('Total Sales ($)')
    ax.legend()
    return fig

st.pyplot(plot_scottsdale_data(df))

# Janesville Data with Median
st.header('Median Price Data for Janesville')
def plot_janesville_data(df):
    janesville_dealerships = df[df['Dealer_Region'] == 'Janesville']
    janesville_data = janesville_dealerships.groupby('Dealer_Name')['Price ($)'].sum()
    median_price = janesville_data.median()
    fig, ax = plt.subplots()
    janesville_data.plot(kind='bar', ax=ax)
    ax.axhline(median_price, color='red', linestyle='--', label='Median Price')
    ax.set_title('Janesville Data with Median')
    ax.set_xlabel('Dealer Name')
    ax.set_ylabel('Total Sales ($)')
    ax.legend()
    return fig

st.pyplot(plot_janesville_data(df))


# Price Prediction Section 
st.markdown("""
## Sales Forecast
The predictive models trained for car price estimation clearly indicate that prices vary when input parameters are altered. These changes confirm a correlation between the selected features and the prices generated by machine learning algorithms. For example:

Gender: A variation in price for the identical car when purchased by a male compared to a female suggests possible differences in negotiation patterns across genders.

Regional Variations: The disparity in prices across different regions likely mirrors local market conditions, including economic status, and supply and demand factors. Dealerships may be setting their prices to align with the purchasing power prevalent in each local market.

""")

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

    st.write("Model: Random Forest Regressor")
    st.write("Mean Absolute Error (MAE): 4456.283793801416 ")
    st.write("R-squared (R^2): 0.6485862780515839")

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
st.header("3D Clustering")
st.markdown("""
### 3D Clustering Visualization
This 3D scatter plot shows clusters of cars based on features like price, brand, and model. It helps visualize market segments and consumer preferences.
""")
@st.cache
def load_df_reduced():
    # Load the DataFrame
    return joblib.load('models/pca/df_reduced.joblib')

# Load the reduced DataFrame
df_reduced = load_df_reduced()

fig = px.scatter_3d(df_reduced, x='PC1', y='PC2', z='PC3', color='Cluster', opacity=0.7, size_max=5, title='3D Clustering')
st.plotly_chart(fig, use_container_width=True)