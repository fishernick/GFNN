import generalPurposeDataExports as gp
import random

def build_random_csv():
  rows = random.randint(1,2000)
  columns = random.randint(2,10)
  headers = []
  for num in range(columns):
    headers.append(num)
  data = []
  for _ in range(rows):
    in_data = []
    for __ in range(columns):
      in_data.append(random.randint(1,500))
    data.append(in_data)
  gp.export_data(file_name='testdata', data_array=data, headers=headers, open_mode='w')