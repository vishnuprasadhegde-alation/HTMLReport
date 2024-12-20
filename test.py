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


# df=pd.read_csv("./results_aggregated.csv")
# # df=df[['Label','Error %','Date']]
# df.set_index('Date', inplace=True)
# # df = df.sort_values(by='Date')
# # df.index.name='Label'

# transposed_csv=df.transpose()
# df.rename_axis('Label', inplace=True)
csv_directory = './csvfiles'
# date_row_index = transposed_csv[transposed_csv['Date'] == '2024-09'].index[0]

# new_header = transposed_csv.iloc[date_row_index]  # Get the row with the date values
# transposed_csv.columns = new_header  # Set the row as the new header
# transposed_csv = transposed_csv.drop(date_row_index) 

df_list = []
columns_to_remove = ['# Samples','Average','Median','90% Line','95% Line','Min','Max','Received KB/sec','Std. Dev.','Error %','Throughput']
labels_to_remove=['Compose-Home','Create Glosary','Document hub','CreateArticle','Open Folder','Open Document','Search','TOTAL','Open Aricle - 20 custom fields']

# Create datafile for prediction of response time release over release 
for file_name in os.listdir(csv_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(csv_directory, file_name)
        file_date = file_name.split('_')[-1].split('.')[0]

        df = pd.read_csv(file_path)
        
        matching_rows = df[df['Label'].isin(labels_to_remove)]

        df.drop(matching_rows.index, inplace=True)

        # df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)

        df=df[['Label','Error %']]

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

combined_df = pd.concat(df_list, ignore_index=True, sort=False)
print(combined_df.head())