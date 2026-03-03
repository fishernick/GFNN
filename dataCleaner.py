import generalPurposeDataExports as gp
import pandas as pd
import shelve
import csv
import os

df = pd.read_csv("stock_prices.csv")
print(df.head())

def drop_column(df, column_name):
  df = df.drop(columns = [column_name])
  return df

def main_cleaner():
  csv_file_name = input("File Name\n")
  if(csv_file_name.split('.')[1] != "csv"):
    csv_file_name = csv_file_name.split('.')[0] + '.csv'
  while