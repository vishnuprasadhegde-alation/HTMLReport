import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template
import os
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


#create historic aggregate report file for prediction
# date,response time,label,errors 

#read from s3


# Read the dataset from CSV
df = pd.read_csv('testdatastock.csv')


# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as index
df.set_index('Date', inplace=True)
df['day_of_week'] = df.index.dayofweek


df.dropna(subset=['AMZN', 'META', 'GOOG', 'ORCL', 'MSFT', 'day_of_week'], inplace=True)

# Define target variables and exogenous variables
target_vars = ['AMZN', 'META', 'GOOG', 'ORCL', 'MSFT']  # List of target variables
exog_vars = ['day_of_week']  # List of exogenous variables


for target in target_vars:
    df[target+'_diff']= df[target].diff(periods=1).diff(periods=30).dropna() 


n_periods = 30 
def fit_sarimax_and_predict(target, exog_vars, df, n_periods=30, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7)):
    # Define the SARIMAX model
    sarimax_model = SARIMAX(df[target],
                            exog=df[exog_vars],
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
    
    # Fit the model
    sarimax_results = sarimax_model.fit()
    print(f"Model summary for {target}:")
    print(sarimax_results.summary())
    
    # Generate exogenous variables for the future period (next 'n_periods' days)
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_periods, freq='D')
    
    # Create exogenous features for future dates (assuming daily frequency)
    future_exog = pd.DataFrame({
        'day_of_week': future_dates.dayofweek,
    }, index=future_dates)
    
    # Make forecast for the future period
    forecast = sarimax_results.get_forecast(steps=n_periods, exog=future_exog)
    
    # Extract predicted values and confidence intervals
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    return forecast_values, conf_int, future_dates


# Create dictionaries to store the forecasts and confidence intervals for all target variables
forecasts = {}
conf_intervals = {}

# Forecast for each target variable
 # Forecasting next 30 days
for target in target_vars:
    forecast_values, conf_int, future_dates = fit_sarimax_and_predict(target+'_diff', exog_vars, df, n_periods=n_periods)
    forecasts[target+'_diff'] = forecast_values
    conf_intervals[target+'_diff'] = conf_int

# Plot the forecasts for the first few target variables (for example, AMZN, META, GOOG)
plt.figure(figsize=(15, 12))

# Loop through the first 3 targets to plot their forecasts
for i, target in enumerate(target_vars[:5]):
    plt.subplot(8, 1, i+1)
    plt.plot(df.index, df[target+'_diff'], label=f'Actual {target}', color='blue')
    plt.plot(future_dates, forecasts[target+'_diff'], label=f'Forecast {target}_diff', color='red')
    plt.fill_between(future_dates, conf_intervals[target+'_diff'].iloc[:, 0], conf_intervals[target+'_diff'].iloc[:, 1], color='pink', alpha=0.3)
    plt.title(f'{target}_diff Forecast using SARIMAX')
    plt.xlabel('Date')
    plt.ylabel(f'{target}_diff Value')
    plt.legend()

plt.tight_layout()
# plt.show()
trends_graph_file = os.path.join("reports", "stock_predictions.png")
plt.savefig(trends_graph_file)
plt.close()

# Optionally, save the forecasts and confidence intervals to CSV
forecast_df = pd.DataFrame(forecasts)
forecast_df.index=future_dates
forecast_df.to_csv('reports/forecasted_values_future.csv')


conf_int_df = pd.concat(conf_intervals.values(),axis=1)
conf_int_df.index=future_dates
conf_int_df.to_csv('reports/confidence_intervals_future.csv')



# Function to generate stock trend graph
def generate_trends_graph(df):
    plt.figure(figsize=(8, 5))
    plt.plot(df["Date"], df["AMZN"], label="AMZN")
    plt.plot(df["Date"], df["META"], label="META")
    plt.plot(df["Date"], df["GOOG"], label="GOOG")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Stock Price Trends")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    trends_graph_file = os.path.join("reports", "stock_trends.png")
    plt.savefig(trends_graph_file)
    plt.close()
    return trends_graph_file



# Load data (replace this with your own data or CSV)
data = {
    "Date": pd.date_range(start="2020-01-01", periods=30),
    "AMZN": [120 + i for i in range(30)],
    "META": [300 + i * 2 for i in range(30)],
    "GOOG": [270 + i * 1.5 for i in range(30)],
}

df = pd.DataFrame(data)

# Generate graphs
generate_trends_graph(df)
# volatility_graph = generate_volatility_graph(df)
# moving_avg_graph = generate_moving_avg(df)

# Render table
table_html = df.to_html(index=False, border=1, classes="table")

# Create text summary
text_summary = """
This report showcases stock trends, volatility insights, and moving averages for AMZN, META, and GOOG over the last 30 days.
Volatility is calculated based on daily price change magnitudes, while moving averages are calculated over a 5-day window.
"""

# Load external HTML template
template_path = os.path.join("templates", "template.html")
with open(template_path, "r") as f:
    template_content = f.read()

template = Template(template_content)

# Render the full HTML
html_content = template.render(
    text_summary=text_summary,
    table=table_html,
    stock_predictions_graph="stock_predictions.png",
    stock_trends_graph="stock_trends.png",  # Relative to the report directory
)


# Save the full HTML report
output_dir = "reports"
output_file = os.path.join(output_dir,"final_stock_report.html")
with open(output_file, "w") as f:
    f.write(html_content)

print(f"HTML report generated and saved to {output_file}")
