import neuralNetBuilder as nn
#import dataCleaner as dc
import art
import sys
import os

if __name__ == "__main__":
  Art=art.text2art("Neural Network")
  print(Art)
  try:
    term_size = os.get_terminal_size()
    columns = term_size.columns
  except OSError:
    columns = 80 
  while True:
    print('-' * columns)
    dec = input("1. Neural Network Program\n2. Data Cleaner\n3. Exit\n")
    match dec:
      case "1":
        print('-' * columns)
        nn.network_builder()
      case "2":
        print('-' * columns)
        #dc.main_cleaner()
      case "3":
        sys.exit()
      case default:
        print("the fuck is wrong with you?")
      
  