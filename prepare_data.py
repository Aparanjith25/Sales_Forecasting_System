import pandas as pd

df = pd.read_csv('train.csv', low_memory=False, usecols=['Store', 'Date', 'Sales', 'Open'])
store1 = df[df['Store'] == 1].copy()
store1['Date'] = pd.to_datetime(store1['Date'])
store1 = store1.sort_values('Date')
store1 = store1[store1['Open'] == 1]
store1 = store1[['Date', 'Sales']]
store1.to_csv('store1_sales.csv', index=False)

print("Done! Created store1_sales.csv")