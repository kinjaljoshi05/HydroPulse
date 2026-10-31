# Reservoir Water Level Forecasting
<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5">
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3">

  

  
</p>


## Problem Statement
### Background
WRI’s updated Aqueduct Water Risk Atlas finds that 17 countries, which are home to a quarter of the world’s population, face “extremely high” water stress. India ranks 13th for overall water stress and has more than three times the population of the other 17 extremely highly stressed countries combined. India’s water challenges extend beyond current events in Chennai. Last year, the National Institution for Transforming India (NITI Aayog), a government research agency, declared that the country is “suffering from the worst water crisis in its history, and millions of lives and livelihoods are under threat.

<img src="img/water-condition.png" alt="Aqueduct Report" style="max-width: 30%; height: auto;">

Cities in the Global South face unreliable, inadequate, and polluted supply of freshwater. About 1 billion people do not have access to safe and continuous (24/7) water supply. These cities show that vast segments of the urban population in the global south lack access to safe, reliable and affordable water. On average, almost half of all households in the studied cities lacked still lack access to piped utility water. 

### Case of Bengaluru, India
Bengaluru, the capital of Karnataka, is one of the fastest growing cities in India. The city is known for its pleasant climate, beautiful gardens, and lakes. However, the city is facing a severe water crisis. According to the experts Bengaluru is on the verge of an **imminent water crisis**, with the city’s water table shrinking massively in just two decades. 

The city could follow Cape Town to become the first Indian city to run out of water unless urgent measures are taken. As the population increases, the demand for water also increases. Water is being sourced from Cauvery river reservoirs to meet Bengaluru's increasing water demands. But, the water supply has become a major challenge  due to various factors like climate change, deforestation, unplanned urbanization, depleting ground water resources due to unprecedented borewell extraction etc.

**Cauvery reservoirs** *(Harangi, Hemavathi, KRS, Kabini)*, which are major source of water for Bengaluru, are already under extremely high water risk  category, indicating drought like situation,  as per the data collected from World Resources Institute Aqueduct tool. 

<img src="img/cauvery-basin.png" alt="Cauvery basin" style="max-width: 40%; height: auto;">

Forecasting water availability accurately is challenging, due to uncertain hydro-meteorological  and other factors of Cauvery river basin.

**The objective of this project is to forecast the water levels in the Cauvery reservoirs, which will help in better water management and planning for the city of Bengaluru.**




## Solution
A multi-stacking ensemble based multi-variate time series analysis model using machine learning, deep learning and statistical procedures like Support Vector Regressor(SVR), XGBoost, Facebook Neural Prophet, Long Short Term Memory(LSTM) and Seasonal Auto Regressive Integrated Moving Averages (SARIMA) with near real time reservoir water level data, and associated hydro-meteorological data like rainfall, temperature etc. for better and accurate temporal water level forecasting.

## Data
The dataset is obtained from the official WRI website and India WRIS datasets. It includes the geospatial data of four cauvery river basin reservoirs Hemavathi, Harangi, Kabini and KRS.  
| Dataset Source | Dataset Name | Date Range               | Original Row Count | Missing Rows | Final Date Range      | Final Row Count |
|---------------|-------------|-------------------------|--------------------|--------------|----------------------|----------------|
| WRI           | Harangi     | 30-09-2010 to 16-12-2020 | 3321               | 332          | 01-01-2011 to 31-12-2020 | 3653           |
| WRI           | Hemavathi   | 30-09-2010 to 16-12-2020 | 3314               | 339          | 01-01-2011 to 31-12-2020 | 3653           |
| WRI           | KRS         | 30-09-2010 to 16-12-2020 | 3313               | 340          | 01-01-2011 to 31-12-2020 | 3653           |
| WRI           | Kabini      | 30-09-2010 to 16-12-2020 | 3314               | 339          | 01-01-2011 to 31-12-2020 | 3653           |


## Functional Requirements
1. Dynamic Dashboard to view the results of the predictions
2. Visualizations of forecasted data
3. Predicting the water level using other features such as Inflow, Outflow, Rainfall, Soil Moisture, Humidity etc.
4. 1-3 months forecasts of water level
5. Correlation of the features crucial of water level prediction
6. Reservoir-wise forecasting
7. Scalable design to add new reservoir forecasting 


## Evaluation Metrics
Root Mean Square Error (RMSE), R<sup>2</sup>, Mean Absolute Percentage Error (MAPE)

## Final Design of the Project
Here are some screenshots of the final design of the project.
<img src="img/dash1.png" alt="Dashboard1" style="max-width: 80%; height: auto;">
<img src="img/dash2.png" alt="Dashboard2" style="max-width: 80%; height: auto;">
<img src="img/dash3.png" alt="Dashboard3" style="max-width: 80%; height: auto;">

## Dashboard Features
Demo video of the dashboard can be found [here](https://drive.google.com/file/d/1sTAAyVX3CmPLgNGGhuxafODhc-3nX2Ki/view?usp=sharing).
- Reservoir wise water forecasting
- Collective water forecasting
- Algorithm wise water forecasting
- Feature toggle options like with/without weather data
- Scalable design to add new reservoir forecasting




