import csv
import torch
import torch.nn as nn
import pandas as pd
import shelve
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np
import random
import math

def prompt(message, default, cast=str):
    value = input(f"{message} (default {default}): ")
    return cast(value) if value else default

def build_tensors(csv_file_name, input_length):
  header = pd.read_csv(f'models/{csv_file_name}.csv', header=None, nrows=1).values.tolist()[0]
  for index, item in enumerate(header):
    header[index] = str(item)
  with open(f'{csv_file_name}.csv') as f:
    rows = list(csv.DictReader(f))
  X = torch.tensor([[float(r[name]) for name in header[:input_length]] for r in rows])[:-50]
  Y = torch.tensor([[float(r[name]) for name in header[(input_length):]] for r in rows])[:-50]
  return [X,Y], len(header) - input_length

def normalize_tensors(X, Y):
    X_min, X_max = X.min(dim=0).values, X.max(dim=0).values
    Y_min, Y_max = Y.min(dim=0).values, Y.max(dim=0).values
    x = (X - X_min) / (X_max - X_min).clamp(min=1e-8)
    y = (Y - Y_min) / (Y_max - Y_min).clamp(min=1e-8)
    return [x, y], (X_min, X_max), (Y_min, Y_max)

def unnormalize_tensor(x, Y_min, Y_max):
    return x * (Y_max - Y_min) + Y_min

def build_model(input_dimensions, output_dimensions, hidden_layers_length, complexity=1.0, reconstruction=False):
  base_dimensionality = 32
  hidden_dimensionality = int(base_dimensionality*complexity)
  input_features = input_dimensions
  layers = []
  used_dimensions = []
  all_dimensions = [input_dimensions]

  for length in range(hidden_layers_length):
    #base case: triangular network
    if (length < int(hidden_layers_length/2) + hidden_layers_length % 2):
      next_dimension = max(int(hidden_dimensionality/hidden_layers_length)*(length+1)*2, 1)
      all_dimensions.append(next_dimension)
      used_dimensions.append(next_dimension)
      layers.append(nn.Linear(input_features, next_dimension))
      layers.append(nn.ReLU())
      input_features = next_dimension
    else:
      if (hidden_layers_length % 2 == 0):
        next_dimension = used_dimensions[len(used_dimensions)-1]
        used_dimensions.pop()
      else:
        next_dimension = used_dimensions[len(used_dimensions)-2]
        used_dimensions.pop()
      all_dimensions.append(next_dimension)
      layers.append(nn.Linear(input_features, next_dimension))
      layers.append(nn.ReLU())
      input_features = next_dimension

  layers.append(nn.Linear(input_features, output_dimensions))
  all_dimensions.append(output_dimensions)
  if not reconstruction:
    print('Creating neural network with neuron pathways:')
    for index, neuron in enumerate(all_dimensions):
      if (index == len(all_dimensions)-1):
         print(f'{neuron}')
      else:
        print(f'{neuron}-', end="")
  return nn.Sequential(*layers)

def training_loop(EPOCHS, model, learning_rate, tensors, device):
  x, y = tensors[0].to(device), tensors[1].to(device)
  best_loss = float('inf')
  patience, patience_counter = 500, 0
  loss_fn   = nn.MSELoss() 
  optimizer = torch.optim.Adam(model.parameters(), learning_rate)
  pbar = tqdm(range(EPOCHS), desc=f'loss: {best_loss}')
  for epoch in pbar:
      model.train()
      predictions = model(x)
      loss = loss_fn(predictions, y)
      optimizer.zero_grad()
      loss.backward()
      if(epoch % 10 == 0):
        pbar.set_description(f'loss: {best_loss}')
        chance = epoch/EPOCHS
      optimizer.step()
      if loss.item() < best_loss - 1e-6:
          best_loss = loss.item()
          patience_counter = 0
      else:
          patience_counter += 1
          if patience_counter >= patience:
              pbar.close()
              chance = epoch/EPOCHS
              print(f"\nEarly stopping at epoch {epoch+1}")
              break
  
  return model, chance

def evaluate(model, tensors, y_norm_params):
  model.eval()
  x, y = tensors[0], tensors[1]
  model.to("cpu")
  Y_min, Y_max = y_norm_params
  with torch.no_grad():
      predictions = model(x)
      loss = nn.MSELoss()(predictions, y)
      print(f"final loss: {loss.item():.8f}")
      predictions_unnorm = unnormalize_tensor(predictions, Y_min, Y_max)
      actual_unnorm = unnormalize_tensor(y, Y_min, Y_max)
      mae = (predictions_unnorm - actual_unnorm).abs().mean()
      print(f"mae (unnormalized): {mae.item():.4f}")
  return predictions_unnorm

def save_model(csv_file_name, model, input_length, output_length, hidden_layers, complexity, tensors):
  # saved as db[name] = [input, output, layers, complexity, tensors]
  with shelve.open('modeldata.db') as db:
     db[csv_file_name] = [input_length, output_length, hidden_layers, complexity, tensors]
  torch.save(model.state_dict(), f'models/{csv_file_name}.pth')

def infer(csv_file_name, data_input):
  with shelve.open('modeldata.db') as db:
    parameters = db[csv_file_name]
  model = build_model(parameters[0],parameters[1],parameters[2],parameters[3], reconstruction=True)
  model.load_state_dict(torch.load(f'models/{csv_file_name}.pth'))
  model.eval()
  tensors = parameters[4]
  _, x_params, y_params = normalize_tensors(*tensors)
  X_min, X_max = x_params
  Y_min, Y_max = y_params
  mse_value = 0
  for data_values in data_input:
    data = pd.read_csv(f'{csv_file_name}.csv').iloc[data_values].values.tolist()
    raw_input = torch.tensor([data[:parameters[0]]], dtype=torch.float32)
    x_norm = (raw_input - X_min) / (X_max - X_min).clamp(min=1e-8)
    with torch.no_grad():
      prediction_norm = model(x_norm)
      prediction = unnormalize_tensor(prediction_norm, Y_min, Y_max)
    predicted_values = prediction.tolist()[0]
    true_values = data[parameters[0]:]
    mse_value = mean_squared_error(predicted_values, true_values)
  return prediction.tolist()[0], data[parameters[0]:], mse_value/len(data_input)

def calculate_error_metrics(csv_file_name, input_length):
  data = pd.read_csv(f'{csv_file_name}.csv')
  average_output_size = np.array(data.iloc[:,input_length:].mean()).mean()
  _, _, mse = infer(csv_file_name, [len(data)-51,len(data)-1])
  print("Error metrics use the last 50 elements as testing values, which are not used in normal testing.")
  print(f'Mean squared error: {mse}\nMse over average output (percent error): {mse/average_output_size}')

def auto_build_metrics(csv_file_name, input_length, tensors, model):
  model.to("cpu")
  model.eval()
  data = pd.read_csv(f'{csv_file_name}.csv').values.tolist()
  data_input = [len(data)-51,len(data)-1]
  _, x_params, y_params = normalize_tensors(*tensors)
  X_min, X_max = x_params
  Y_min, Y_max = y_params
  for data_values in data_input:
    data = pd.read_csv(f'{csv_file_name}.csv').iloc[data_values].values.tolist()
    raw_input = torch.tensor([data[:input_length]], dtype=torch.float32)
    x_norm = (raw_input - X_min) / (X_max - X_min).clamp(min=1e-8)
    with torch.no_grad():
      prediction_norm = model(x_norm)
      prediction = unnormalize_tensor(prediction_norm, Y_min, Y_max)
    predicted_values = prediction.tolist()[0]
    true_values = data[input_length:]
    mse = mean_squared_error(predicted_values, true_values)
  data = pd.read_csv(f'{csv_file_name}.csv')
  average_output_size = np.array(data.iloc[:,input_length:].mean()).mean()
  return mse/average_output_size


def auto_build(csv_file_name, input_length, error_metric=0.01, max_attempts=10, epochs=20000):
    complexity = 2.0
    best_model = None
    best_error = float('inf')
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        hidden_layers = max(2, int(math.sqrt(complexity) + complexity / 2))
        print(f"\nattempt {attempt}: complexity={complexity:.2f}, hidden_layers={hidden_layers}")
        model, tensors, chance = build_net(
            csv_file_name, input_length,
            hidden_layers=hidden_layers,
            complexity=complexity / 2,
            epochs=epochs,
            auto=True
        )
        error = auto_build_metrics(csv_file_name, input_length, tensors, model)
        print(f"error ratio: {error:.4f} (target: {error_metric})")
        if error < best_error:
            best_error = error
            best_model = (model, tensors, hidden_layers, complexity)

        if error <= error_metric:
            print(f"target reached on attempt {attempt}")
            break
        scale = min(error / error_metric, 2.0)
        complexity = min(complexity * scale, 64.0)
        if chance < 0.3:
            epochs = int(epochs * 1.5)

    if best_model is None:
        print("auto build failed to find a suitable model.")
        return None

    model, tensors, hidden_layers, complexity = best_model
    _, output_length = build_tensors(csv_file_name, input_length)
    save_model(csv_file_name, model, input_length, output_length, hidden_layers, complexity / 2, tensors)
    print(f"saved best model with error ratio: {best_error:.4f}")
    return model
  

"""
input_length will be the input neurons, the output neurons will 
be the following headers within the provided csv.
"""
def build_net(csv_file_name, input_length, hidden_layers=5, complexity=1.0, learning_rate=0.001, epochs=20000, auto=False):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f'Processing using: {device}')
  tensors, output_length = build_tensors(csv_file_name, input_length)
  normal_tensors, x_params, y_params = normalize_tensors(*tensors)
  model = build_model(input_length, output_length, hidden_layers, complexity).to(device) 
  model, chance = training_loop(epochs, model, learning_rate, normal_tensors, device)
  if not auto:
    evaluate(model, normal_tensors, y_params)
    save_model(csv_file_name, model, input_length, output_length, hidden_layers, complexity, tensors)
  else:
    return model, tensors, chance
  
def network_builder():
  while True:
    selection = input("1. Network Builder\n2. Inference Model\n3. Calculate Error Metrics\n4. Exit\n")
    match(selection):
      case "1":
        csv_file_name = prompt("csv file name", None)
        input_length  = prompt("input length", None, int)
        auto_building = prompt("enable auto building", None, bool)
        if not auto_building:
          hidden_layers = prompt("hidden layers", 5, int)
          complexity    = prompt("complexity", 1.0, float)
          learning_rate = prompt("learning rate", 0.001, float)
          epochs        = prompt("epochs", 20000, int)
          build_net(csv_file_name, input_length, hidden_layers, complexity, learning_rate, epochs)
        else:
          auto_build(csv_file_name, input_length)
      case "2":
        name = prompt("model name", None)
        data = prompt("data row", None, int)
        inferred_value, real_value, mse = infer(name, [data])
        print(f'inferred_value: {inferred_value}\nreal_value: {real_value}\nmse: {mse}')
      case "3":
        name = prompt("model name", None)
        with shelve.open('modeldata.db') as db:
          input_length = db[name][0]
        calculate_error_metrics(name,input_length)
      case "4":
        return


