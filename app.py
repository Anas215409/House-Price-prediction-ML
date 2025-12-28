import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor


st.title('🏠House Price prediction using ML')

st.image('https://media.tenor.com/B9G3PV_TOBYAAAAM/happy-birthday.gif')

df = pd.read_csv('house_data.csv')

X = df.iloc[:,:-3]
y = df.iloc[:,-1]

st.sidebar.title('🏠Select house features')
st.sidebar.image('https://media.tenor.com/B9G3PV_TOBYAAAAM/happy-birthday.gif')
all_value = []
for i in X:
  min_value = int(X[i].min())
  max_value = int(X[i].max())
  ans = st.sidebar.slider(f'Select {i} value', min_value, max_value)
  all_value.append(ans)

# st.write(all_value)

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

final_value = scaler.transform([all_value])

model = RandomForestRegressor()
model.fit(X,y)
house_price = model.predict(final_value)

with st.spinner('predicting house price'):
  time.sleep(1)
msg = f'''House Price is: ${house_price*100000}'''
st.success(msg)

st.markdown('''**Design and Developed by: Anas**''')





