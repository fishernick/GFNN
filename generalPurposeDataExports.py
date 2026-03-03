"""
General purpose data collection module
"""
import csv
import pandas as pd
import os

"""
general purpose data exporting function
supports 1d and 2d arrays
"""
def export_data(file_name, data_array, headers="", open_mode='a'):
  with open(f'{file_name}.csv', open_mode, newline='') as csv_file:
    file_writer = csv.writer(csv_file)
    try:
      header = pd.read_csv(f'{file_name}.csv', header=None, nrows=1).values.tolist()[0]
      #single append
      if(type(data_array[0]) == type("a")):
        data_length = len(data_array)
      else:
        data_length = len(data_array[0])
      if (len(header) != data_length):
        print(header)
        print(data_length)
        print('File header length doesn\'t match input data. \nExiting for data safety.')
        return
    except (pd.errors.EmptyDataError):
      file_writer.writerow(headers)
    if(type(data_array[0]) == type("a")):
      file_writer.writerow(data_array)
      csv_file.flush()
    else:
      for row in data_array:
        file_writer.writerow(row)
        csv_file.flush()

"""
just deletes a csv file from the cwd by name... js stick it in vro 
"""
def clear_data_file(file_name):
  if os.path.exists(f'{file_name}.csv'):
    os.remove(f'{file_name}.csv')
  else:
    print(f"File not found")