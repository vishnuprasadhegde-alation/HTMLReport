import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template
import os
import glob
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime

prediction_period=2
# Directory containing CSV files
csv_directory = './csvfiles'

# List to store individual DataFrames
df_list = []
columns_to_remove = ['# Samples','Average','Median','90% Line','95% Line','Min','Max','Received KB/sec','Std. Dev.','Error %','Throughput']
labels_to_remove=['Compose-Home','Create Glosary','Document hub','CreateArticle','Open Folder','Open Document','TOTAL','Open Aricle - 20 custom fields']

# Create datafile for prediction of response time release over release 
for file_name in os.listdir(csv_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(csv_directory, file_name)
        file_date = file_name.split('_')[-1].split('.')[0]

        df = pd.read_csv(file_path)
        
        matching_rows = df[df['Label'].isin(labels_to_remove)]

        df.drop(matching_rows.index, inplace=True)

        df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)
        print("without transpose:", df.head)

        transposed_csv=df.T
        new_header = transposed_csv.iloc[0]
        transposed_csv = transposed_csv[1:]
        transposed_csv.columns = new_header
        # transposed_csv['Date'] = file_date
        # transposed_csv.set_index('Date', inplace=True)
        # print("transfromed datafsre :", transposed_csv.head())

        transposed_csv['Date'] = file_date
        df_list.append(transposed_csv)
        


# Concatenate all DataFrames into one
combined_df = pd.concat(df_list, ignore_index=True, sort=False)

# Save combined DataFrame to a new CSV file
combined_df.to_csv('./responsetime_aggregated.csv', index=False)
print("responsetime_aggregated CSV file created successfully with Date column!")


#Create datafile for KPI graphs 
df1_list=[]
for file_name in os.listdir(csv_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(csv_directory, file_name)
        file_date = file_name.split('_')[-1].split('.')[0]

        df = pd.read_csv(file_path)
        # df1=df['Label','Error %']
        # df1 = pd.DataFrame().assign(Label=df['Label'], Errors=df['Error %'])

        matching_rows = df[df['Label'].isin(labels_to_remove)]

        df.drop(matching_rows.index, inplace=True)

        # transposed_csv=df1.T
        # new_header = transposed_csv.iloc[0]
        # transposed_csv = transposed_csv[1:]
        # transposed_csv.columns = new_header
        # transposed_csv['Date'] = file_date
        # transposed_csv.set_index('Date', inplace=True)
        # print("transfromed datafsre :", transposed_csv.head())

        df['Date'] = file_date
        df1_list.append(df)

combined_df1 = pd.concat(df1_list, ignore_index=True, sort=False)
combined_df1.to_csv('./results_aggregated.csv', index=False)
print("results_aggregated CSV file created successfully with Date column!")



# Use glob to get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(csv_directory, '*.csv'))
latest_file=""
# If there are no CSV files, print a message and exit
if not csv_files:
    print("No CSV files found in the result file directory.")
else:
    # Sort the CSV files by modification time (latest first)
    latest_file = max(csv_files, key=os.path.getmtime)
    print(f"The latest CSV file is: {latest_file}")

    # function call for graph and table for latest release

# Read the dataset from CSV
df = pd.read_csv('responsetime_aggregated.csv')

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Set 'Date' as index
df.set_index('Date', inplace=True)
df['day_of_week'] = df.index.dayofweek


df.dropna(subset=['Tag', 'Custom_Fields', 'ReadColumnAPI', 'Update column values API', 'Login','datasources','Schema Selection','Table selection','Column Selection','Logout', 'day_of_week'], inplace=True)

# Define target variables and exogenous variables
# target_vars = ['Tag', 'Custom_Fields', 'ReadColumnAPI', 'Update column values API', 'Login','datasources','Schema Selection','Table selection','Column Selection','Logout']  # List of target variables
target_vars = ['Login','Homepage','Search','datasources','Schema Selection','Table selection','Column Selection','Logout','Tag','Custom_Fields', 'ReadColumnAPI', 'Update column values API']  # List of target variables

exog_vars = ['day_of_week']  # List of exogenous variables


for target in target_vars:
    df[target+'_diff']= df[target].diff(periods=1).dropna() 
print("differentiated: " , df.head)
df.to_csv("differentiated.csv")


def fit_sarimax_and_predict(target, exog_vars, df, n_periods=prediction_period, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7)):
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
    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=prediction_period, freq='MS')
    
    # Create exogenous features for future dates (assuming daily frequency)
    future_exog = pd.DataFrame({
        'day_of_week': future_dates.month,
    }, index=future_dates)
    
    # Make forecast for the future period
    forecast = sarimax_results.get_forecast(steps=prediction_period, exog=future_exog)
    
    # Extract predicted values and confidence intervals
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()
    return forecast_values, conf_int, future_dates


# Create dictionaries to store the forecasts and confidence intervals for all target variables
forecasts = {}
conf_intervals = {}

# Forecast for each target variable
 # Forecasting next prediction_period days
for target in target_vars:
    forecast_values, conf_int, future_dates = fit_sarimax_and_predict(target+'_diff', exog_vars, df, n_periods=prediction_period)
    last_value = df[target].iloc[-1]
    forecasts[target+'_diff'] = forecast_values.cumsum()+last_value
    conf_intervals[target+'_diff'] = conf_int+last_value

# Plot the forecasts for the first few target variables (for example, AMZN, META, GOOG)
plt.figure(figsize=(15, 40))

# Loop through the first 3 targets to plot their forecasts
for i, target in enumerate(target_vars[:12]):
    plt.subplot(12, 1, i+1)
    plt.plot(df.index, df[target], label=f'Actual {target}', color='blue')
    plt.plot(future_dates, forecasts[target+'_diff'], label=f'Forecast {target}', color='red')

    last_actual_date = df[target].index[-1]
    first_predicted_date = forecasts[target+'_diff'].index[0]

    # Get the values corresponding to these dates
    y_actual_value = df[target][-1]  # Last actual value
    y_pred_value = forecasts[target+'_diff'][0]  # First predicted value

    # Add a vertical dotted line between the last actual value and the first predicted value
    plt.plot([last_actual_date, first_predicted_date], [y_actual_value, y_pred_value], color='blue', lw=1)

    plt.fill_between(future_dates, conf_intervals[target+'_diff'].iloc[:, 0], conf_intervals[target+'_diff'].iloc[:, 1], color='pink', alpha=0.3)
    plt.title(f'{target} ')
    plt.xlabel('Date')
    plt.ylabel(f'Response Time\n in milliseconds', labelpad=70,rotation=0)
    plt.legend()

plt.tight_layout()
# plt.show()
trends_graph_file = os.path.join("reports", "responsetime_predictions.png")
plt.savefig(trends_graph_file)
plt.close()

# Optionally, save the forecasts and confidence intervals to CSV
forecast_df = pd.DataFrame(forecasts)
forecast_df.index=future_dates
forecast_df.to_csv('reports/forecasted_values_future.csv')


conf_int_df = pd.concat(conf_intervals.values(),axis=1)
conf_int_df.index=future_dates
conf_int_df.to_csv('reports/confidence_intervals_future.csv')



# Function to generate responsetime trend graph
df_responseTime_trend_graph=pd.read_csv("./responsetime_aggregated.csv")
df_responseTime_trend_graph=df_responseTime_trend_graph.sort_values(by='Date')
def generate_trends_graph(df_responseTime_trend_graph):
    plt.figure(figsize=(15, 8))
    for i, target in enumerate(target_vars[:12]):
        plt.plot(df_responseTime_trend_graph["Date"], df_responseTime_trend_graph[target], label=f'{target}')
    plt.xlabel("Date")
    plt.ylabel("Response Time(99th Percentile)\n in milliseconds", labelpad=90,rotation=0)
    plt.title("Response Time Trends")
    plt.legend(
        loc='upper right',  # Anchor the legend's upper-left corner
        bbox_to_anchor=(-0.2, 1),  # Position the legend just outside the plot (top-right)
        borderaxespad=0.1  # Optional: adds a small padding between the plot and the legend
)
    plt.xticks(rotation=45)
    plt.tight_layout()
    trends_graph_file = os.path.join("reports", "Responsetime_trends.png")
    plt.savefig(trends_graph_file)
    plt.close()
    return trends_graph_file



# Function to generate Error trend graph
error_list=[]
for file_name in os.listdir(csv_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(csv_directory, file_name)
        file_date = file_name.split('_')[-1].split('.')[0]
        df = pd.read_csv(file_path)        
        matching_rows = df[df['Label'].isin(labels_to_remove)]
        df.drop(matching_rows.index, inplace=True)
        df=df[['Label','Error %']]
        transposed_csv=df.T
        new_header = transposed_csv.iloc[0]
        transposed_csv = transposed_csv[1:]
        transposed_csv.columns = new_header
        transposed_csv['Date'] = file_date
        error_list.append(transposed_csv)


df_errors_trend_graph=pd.concat(error_list, ignore_index=True, sort=False)
df_errors_trend_graph.fillna(0)
df_errors_trend_graph = df_errors_trend_graph.sort_values(by='Date')

def generate_trends_graph1(df_errors_trend_graph):
    plt.figure(figsize=(15, 8))
    for i, target in enumerate(target_vars[:12]):
        plt.plot(df_errors_trend_graph["Date"], df_errors_trend_graph[target], label=f'{target}')
    plt.xlabel("Date")
    plt.ylabel("Errors (Percentage)",labelpad=90,rotation=0)
    plt.title("Errors Trends")
    plt.legend(
        loc='upper right',  # Anchor the legend's upper-left corner
        bbox_to_anchor=(-0.2, 1),  # Position the legend just outside the plot (top-right)
        borderaxespad=0.1  # Optional: adds a small padding between the plot and the legend
)
    plt.xticks(rotation=45)
    plt.tight_layout()
    trends_graph_file = os.path.join("reports", "Errors_trends.png")
    plt.savefig(trends_graph_file)
    plt.close()
    return trends_graph_file


# Function to generate Throughput trend graph
tps_list=[]
for file_name in os.listdir(csv_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(csv_directory, file_name)
        file_date = file_name.split('_')[-1].split('.')[0]
        df = pd.read_csv(file_path)        
        matching_rows = df[df['Label'].isin(labels_to_remove)]
        df.drop(matching_rows.index, inplace=True)
        df=df[['Label','Throughput']]
        transposed_csv=df.T
        new_header = transposed_csv.iloc[0]
        transposed_csv = transposed_csv[1:]
        transposed_csv.columns = new_header
        transposed_csv['Date'] = file_date
        tps_list.append(transposed_csv)


df_tps_trend_graph=pd.concat(tps_list, ignore_index=True, sort=False)
df_tps_trend_graph.fillna(0)
df_tps_trend_graph = df_tps_trend_graph.sort_values(by='Date')

def generate_trends_graph2(df_tps_trend_graph):
    plt.figure(figsize=(15, 8))
    for i, target in enumerate(target_vars[:12]):
        plt.plot(df_tps_trend_graph["Date"], df_tps_trend_graph[target], label=f'{target}')
    plt.xlabel("Date")
    plt.ylabel("Throughput (Requests/seconds)",labelpad=90,rotation=0)
    plt.title("Throughput Trends")
    plt.legend(
        loc='upper right',  # Anchor the legend's upper-left corner
        bbox_to_anchor=(-0.2, 1),  # Position the legend just outside the plot (top-right)
        borderaxespad=0.1  # Optional: adds a small padding between the plot and the legend
)
    plt.xticks(rotation=45)
    plt.tight_layout()
    trends_graph_file = os.path.join("reports", "Throughput_trends.png")
    plt.savefig(trends_graph_file)
    plt.close()
    return trends_graph_file


# Load data (replace this with your own data or CSV)
# data = {
#     "Date": pd.date_range(start="2024-12", periods=30),
#     "AMZN": [120 + i for i in range(30)],
#     "META": [300 + i * 2 for i in range(30)],
#     "GOOG": [270 + i * 1.5 for i in range(30)],
# }

#Code for printing table for latest numbers in html report
df_latest = pd.read_csv(latest_file)

#Render table for latest numbers
table_html = df_latest.to_html(index=False, border=1, classes="table")

#Code for printing component level comparison in html report
df=pd.read_csv('./responsetime_aggregated.csv')
df.set_index('Date', inplace=True)
df = df.sort_values(by='Date')
df.index.name='Transaction Name'
df_component=df.T
print(df_component.head())

# Render table for component level comparison numbers
table_html_component = df_component.to_html(index=True, border=1, classes="table")

# df = pd.DataFrame(data)

# Generate graphs
responseTime_trend_graph=generate_trends_graph(df_responseTime_trend_graph)
errors_trend_graph=generate_trends_graph1(df_errors_trend_graph)
throughput_trend_graph=generate_trends_graph2(df_tps_trend_graph)

# volatility_graph = generate_volatility_graph(df)
# moving_avg_graph = generate_moving_avg(df)


# Create text summary
text_summary = """
This report showcases Transaction level Response times trends & Predictions, Error, Throughput trends for the components of Alation application. We have considered the last five releases test results for the analysis and future two month response times are predicted. 
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
    table_component=table_html_component,
    responseTime_predictions_graph="reports/responsetime_predictions.png",
    responseTime_trends_graph="reports/Responsetime_trends.png",
    errors_trend_graph="reports/Errors_trends.png",
    throughput_trend_graph="reports/Throughput_trends.png"    # Relative to the report directory
)

# Save the full HTML report
output_file = "pyperfmax_report.html"
with open(output_file, "w") as f:
    f.write(html_content)

print(f"HTML report generated and saved to {output_file}")
