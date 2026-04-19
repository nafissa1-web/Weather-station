#1: import libraries
import requests 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.metrics import mean_squared_error 
from datetime import datetime, timedelta 
import pytz #lib for time zone

#2. Fetch current weather data
API_KEY = '195f1783d83d96a46b04ffc3be2031e5'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric" 
    response = requests.get(url) 
    data = response.json() 
    return{
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'humidity': round(data['main']['humidity']),
        'description': data['main'][0]['description'],
        'country': data['IT']['country'],
        'WindGustDir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'WindGustSpeed': data['wind']['speed'],

    }

#3.Read historical data
def read_historical_data(filename):
  df = pd.read_csv(filename) #upload csv file into data frame
  df = pd.dropna() #removes missing values raws
  df = pd.drop_duplicates() #removes duplicate rows
  return df

#4.prepare data for training
def prepare_data(data):
  le = LabelEncoder() 
  data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
  data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
  
  X = data[['MinTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']] # Corrected: Removed duplicate 'MinTemp'
  y = data['RainTomorrow'] 
  return X, y, le

#5.Train Rain Prediction model

def train_rain_model(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #20% for testing. randm state ensures we split same nbr each time
  model = RandomForestClassifier(n_estimators=100, random_state= 42) #n_estimators=100 means we used 100 decision tree
  model.fit(X_train, y_train)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  print('Mean Squared Error is :')
  print(mean_squared_error(y_test, y_pred))
  return model 

#6.Prepare regression data

def prepare_regression_data(data, feature):
  x, y = [], [] 
  for i in range(len(data) - 1): 
      x.append(data[feature].iloc[i])
      y.append(data[feature].iloc[i+1])
  x = np.array(x).reshape(-1, 1) 
  y = np.array(y) 
  return x, y

#7.Train regression data
def train_regression_model(x, y):   
  modal = RandomForestRegressor(n_estimators=100, random_state=42) #100 trees
  modal.fit(x, y)
  return modal

#8.Future prediction
def predict_future(modal, current_value):
  predictions = [current_value]
  for i in range(5):
    next_value = modal.predict(np.array(predictions[-1]).reshape(1, -1))
    predictions.append(next_value[0])
  return predictions[1:]

#9.Weather Analysis Function

API_KEY = '195f1783d83d96a46b04ffc3be2031e5'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code != 200:
        print(f"Error fetching data for {city}: {data.get('message', 'Unknown error')}")
        return None

    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'], 
        'country': data['sys']['country'], 
        'WindGustDir': data['wind']['deg'],
        'pressure': data['main']['pressure'] if 'pressure' in data['main'] else None, 
        'WindGustSpeed': data['wind']['speed'] if 'speed' in data['wind'] else None, 
    }

def read_historical_data(filename):
  df = pd.read_csv(filename) 
  df = df.dropna() 
  df = df.drop_duplicates() 
  return df

def wather_view(): 
  city = input('Enter city name: ')
  current_weather = get_current_weather(city)

  if current_weather is None:
      return 

  
  historical_data= read_historical_data('/content/weather.csv') 
  x, y, le = prepare_data(historical_data)
  rain_model = train_rain_model(x, y)

  wind_deg = current_weather['WindGustDir'] % 360 if current_weather['WindGustDir'] is not None else 0 # Handle None for WindGustDir
  compass_points = [
      ("N", 0, 11.25), ('NNE', 11.25, 33.75), ('NE',33.75, 56.25),
      ('ENE', 56.25, 78.75), ('E', 78.75, 101.25), ('ESE', 101.25, 123.75),
       ('SE', 123.75, 146.25), ('SSE', 146.25, 168.75), ('S', 168.75, 191.25),
      ('SSW', 191.25, 213.75), ('SW', 213.75, 236.25), ('WSW', 236.25, 258.75),
      ('W', 258.75, 281.25), ('WNW', 281.25, 303.75), ('NW', 303.75, 326.25),
      ('NNW', 326.25,348.75)
  ]
  compass_direction = next((point for point, start, end in compass_points if start<= wind_deg <end), "N") # Default to N if not found
  compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1 # If not in classes, assign -1 or handle as appropriate

  feature_cols_for_prediction = ['MinTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']

  current_data_dict = {
      'MinTemp' : current_weather['temp_min'],
      'WindGustDir' : compass_direction_encoded,
      'WindGustSpeed': current_weather['WindGustSpeed'] if current_weather['WindGustSpeed'] is not None else 0, # Handle None
      'Humidity': current_weather['humidity'],
      'Pressure': current_weather['pressure'] if current_weather['pressure'] is not None else 1000, # Handle None, provide a default
      'Temp': current_weather['current_temp']
  }

  current_df = pd.DataFrame([current_data_dict], columns=feature_cols_for_prediction)


  rain_prediction = rain_model.predict(current_df)[0]

  temp_x, temp_y = prepare_regression_data(historical_data, 'Temp')
  temp_model = train_regression_model(temp_x, temp_y)
  hum_x, hum_y = prepare_regression_data(historical_data, 'Humidity')
  hum_model = train_regression_model(hum_x, hum_y)

  future_temp = predict_future(temp_model, current_weather['temp_min'])
  future_hum = predict_future(hum_model, current_weather['humidity'])


  timezone = pytz.timezone('Asia/Karachi')
  now = datetime.now(timezone)
  next_hour = now + timedelta(hours=1)
  next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

  future_times = [(next_hour + timedelta(hours=i)).strftime('%H:00') for i in range(5)] #next 5 times


  print(f"City: {city}, {current_weather['country']}")
  print(f"Feels Like: {current_weather['feels_like']}°C")
  print(f"Current Temperature: {current_weather['current_temp']}°C")
  print(f"Minimum Temperature: {current_weather['temp_min']}°C")
  print(f"Maximum Temperature: {current_weather['current_temp']}°C")
  print(f"Humidity: {current_weather['humidity']}%")
  print(f"Weather Prediction: {current_weather['description']}")
  print(f"Rain Prediction: {'Yes' if rain_prediction else 'No'}")

  print('\nFuture Temperature Predictions')

  for time, temp in zip(future_times, future_temp):
     print(f'{time}: {round(temp,1)}°C')

  print('\nFuture Humidity Predictions')
  for time, humidity in zip(future_times, future_hum):
     print(f'{time}: {round(humidity,1)}%')
wather_view()

