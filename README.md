# Business Intelligence Exam Project: Analytical Insights for Startup Car Dealership 

## Objective
To gain valuable insights into the car market for a group of entrepreneurs planning to open a car dealership. The aim is to understand market trends, consumer preferences, and potential regional opportunities to ensure a successful business launch.

## Problem Statement
For new entrants in the car dealership industry, it’s crucial to understand the dynamics of the car market. Insights into consumer preferences, regional market potentials, and seasonal sales trends are essential for developing effective marketing strategies and choosing the right location for the dealership.

## Tasks
### Market Research and Data Analysis
- Conduct thorough research to identify current trends and preferences in the car market.
- Analyze factors crucial to car buyers, such as price, brand reputation, and other key influences on purchasing decisions.
- Explore regional market potentials to determine the most lucrative location for the dealership.
- Investigate how  and if annual income influences consumer choices in car class and price range.
- Study sales trends throughout the year to identify high seasons for car sales.

### Hypothesis Testing
- Test the hypothesis that there is no significant correlation between car features, brand reputation, and purchasing preferences.
- Investigate whether sales trends are consistent throughout the year without significant seasonal variations.
- Explore if preferences for cars differ between genders

### Data-Driven Strategy Development
- Based on the findings, propose marketing strategies targeting identified consumer segments.
- Recommend a geographic location for the new dealership based on potential market growth and competition.
- Suggest car brands and models to stock that align with consumer preferences and regional demands.

### Application 
- Create an interactive visualization of the findings and solution to problem statement.

_________________________________________________________________________________________________________________________________________________________________________________

## Implementation Instructions

- Clone the repository with ```git clone ```
- Open the repository in VS Code 
- Open the terminal and type ``` source biExam_project-env/bin/activate ``` to activate the virtual environment
- You can now run the notebook with the virtual environment
- To see the data visualization app, type in ```streamlit run app.py ```


## Data Source 

Our dataset is a sales report of car sales. The dataset comprises records of car sales transactions, with details such as the date of sale, customer information including gender and annual income, dealership details, and car specifications including make, model, engine type, transmission, color, and body style. Each entry in the dataset represents a unique car sale, providing insights into customer preferences, pricing trends, and dealership performance across different regions

#### Dataset dictionary

| Column Name    | Description                                            |
|----------------|--------------------------------------------------------|
| Car_id         | Unique identifier for each car sold.                   |
| Date           | Date of the transaction.                               |
| Customer Name  | Name of the customer who purchased the car.            |
| Gender         | Gender of the customer.                                |
| Annual Income  | Annual income of the customer.                         |
| Dealer_Name    | Name of the dealership where the transaction occurred. |
| Company        | The company or brand of the car (e.g., Ford, Dodge).   |
| Model          | The specific model of the car (e.g., Expedition).      |
| Engine         | Description of the car's engine (e.g., Double Overhead Camshaft). |
| Transmission   | Type of transmission (e.g., Manual, Auto).            |
| Color          | Color of the car.                                      |
| Price ($)      | Price of the car in dollars.                          |
| Dealer_No      | Dealer number.                           |
| Body Style     | Style or type of the car's body (e.g., SUV, Sedan).    |
| Phone          | Phone number of the dealership.                        |
| Dealer_Region  | Region where the dealership is located.                |


Link to the data source:  https://www.kaggle.com/datasets/missionjee/car-sales-report
