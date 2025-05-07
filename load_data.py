import pandas as pd

def load_data():
  crash_data = pd.read_csv('crash_data.csv')
  return crash_data