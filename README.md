# SPACEX - DATA SCIENCE AND MACHINE LEARNING PROJECT

# ðð¥ðð¬ð¬ð¢ðð² ðð®ðððð¬ð¬ðð®ð¥ ððð§ðð¢ð§ð  ð¨ð ððððð-ð

ððð ðð¢ð§ðð¥ ðð¬ð¬ð¢ð ð§ð¦ðð§ð­ ðð«ð¨ð£ððð­

ðð§ð­ð«ð¨ðð®ðð­ð¢ð¨ð§

â¢ ðð¨ð¥ð¥ððð­ðð ððð­ð ðð«ð¨ð¦ ð©ð®ðð¥ð¢ð ðð©ðððð ððð 
ðð§ð ðð©ðððð ðð¢ð¤ð¢ð©ððð¢ð ð©ðð ð.Created labels column âclassâ which 
classifies successful landings. Explored data using SQL, visualization, folium maps, a
nd dashboards. Gathered relevant columns to be used as features. Changed all cat
egorical variables to binary using one 
hot encoding. Standardized data and used GridSearchCV to find best parameters for machine learning models. 
Visualize accuracy score of all models.

â¢ ðð¨ð®ð« ð¦ððð¡ð¢ð§ð ð¥ððð«ð§ð¢ð§ð  ð¦ð¨ððð¥ð¬ ð°ðð«ð ð©ð«ð¨ðð®ððð:
Logistic Regression, Support Vector Machine, Decision Tree Classifier, and K Nearest 
Neighbors. All produced similar results with ðððð®ð«ððð² ð«ðð­ð ð¨ð ððð¨ð®ð­ ðð.ðð%. All 
models over predicted successful landings. More data is needed for better model determination and accuracy.

# METHODOLOGY

â¢ ððð­ð ðð¨ð¥ð¥ððð­ð¢ð¨ð§ ð¦ðð­ð¡ð¨ðð¨ð¥ð¨ð ð²:

â¢ Combined data from SpaceX public API and SpaceX 
Wikipedia page

â¢ ððð«ðð¨ð«ð¦ ððð­ð ð°ð«ðð§ð ð¥ð¢ð§ð 

â¢ Classifying true 
landings as successful and unsuccessful otherwise
â¢ Perform exploratory data analysis(EDA) using visualization and SQL
â¢ Perform interactive visual analytics using ðð¨ð¥ð¢ð®ð¦ and ðð¥ð¨ð­ð¥ð² ððð¬ð¡
â¢ Perform predictive analysis using classificationmodels
â¢ Tuned models using ðð«ð¢ððððð«ðð¡ðð

# Data Collection

â¢ Data collection process involved a combination of API requests from Space X public API 
and web scraping data from a table in Space Xâs Wikipedia entry.
â¢ The next slide will show the flowchart of data collection from API and 
the one after will show the flowchart of data collection from webscraping.
â¢ Space X API Data Columns:
â¢ FlightNumber, Date, BoosterVersion, PayloadMass, Orbit, 
LaunchSite, Outcome, Flights, GridFins,
â¢ Reused, Legs, LandingPad, Block, ReusedCount, Serial, Longitude, Latitude
â¢ Wikipedia Webscrape Data Columns:
â¢ Flight No., Launch site, Payload, PayloadMass, Orbit, Customer, Launch outcome,
Version Booster, Booster landing, Date, Time


# Data Collection â SpaceX API

https://github.com/YashMewati/SpaceX---Data-Science-and-Machine-Learning-Project/blob/main/Week1%20_%20SpaceX%20Falcon%20Data%20Collection-Wrangling.ipynb

# Data Collection - Scraping

https://github.com/YashMewati/SpaceX---Data-Science-and-Machine-Learning-Project/blob/main/Web%20scraping%20Falcon%209%20and%20Falcon%20Heavy%20Launches%20Records%20from%20Wikipedia.ipynb

# EDA 

â¢ Exploratory Data Analysis performed on 
variables Flight Number, Payload Mass, Launch Site, Orbit, Class and Year.

â¢ ðð¥ð¨ð­ð¬ ðð¬ðð:

â¢ Flight Number vs. Payload Mass, Flight Number vs. Launch Site, Payload Mass
vs. Launch Site, Orbit vs. Success Rate, Flight Number vs. Orbit, Payload vs Orb
it, and Success Yearly Trend
â¢ Scatter plots, line charts, and bar plots were used to compare relationships 
between variables to
â¢ decide if a relationship exists so that they could be used in training the 
machine learning model

# Flight Number vs. Launch Site

â¢ ðð«ððð§ ð¢ð§ðð¢ððð­ðð¬ ð¬ð®ðððð¬ð¬ðð®ð¥ ð¥ðð®ð§ðð¡; ðð®ð«ð©ð¥ð ð¢ð§ðð¢ððð­ðð¬ ð®ð§ð¬ð®ðððð¬ð¬ðð®ð¥ ð¥ðð®ð§ðð¡.

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_16_18 PM](https://user-images.githubusercontent.com/85125898/150124039-f94eca1a-3335-431a-ba19-c519e1b8fb5f.png)

# Payload vs. Launch Site

â¢ ðð«ððð§ ð¢ð§ðð¢ððð­ðð¬ ð¬ð®ðððð¬ð¬ðð®ð¥ ð¥ðð®ð§ðð¡; ðð®ð«ð©ð¥ð ð¢ð§ðð¢ððð­ðð¬ ð®ð§ð¬ð®ðððð¬ð¬ðð®ð¥ ð¥ðð®ð§ðð¡.

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_19_44 PM](https://user-images.githubusercontent.com/85125898/150124344-c2ee0d8c-877f-46e8-a03b-b2362cc3d0c8.png)

# Success Rate vs. Orbit Type

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_21_27 PM](https://user-images.githubusercontent.com/85125898/150127010-de5c1fdb-876d-483e-a965-0bfc71749785.png)

â¢ Success Rate Scale with 0 as 0%
â¢ 0.6 as 60% 1 as 100%
â¢ ES-L1 (1), GEO(1), HEO (1) have 100% success rate (sample sizesin parenthesis) SSO(5) has 100% success rate
â¢ VLEO (14) has decentsuccess rate and attempts
â¢ SO (1) has 0% success rate
â¢ GTO (27) has the around 50% success rate butlargestsampl

# Launch Success Yearly Trend

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_23_48 PM](https://user-images.githubusercontent.com/85125898/150127115-643d76af-3c4a-4b2f-8b22-b4114f2d5db5.png)

â¢ ðð% ðð¨ð§ðð¢ððð§ðð ð¢ð§ð­ðð«ð¯ðð¥ (ð¥ð¢ð ð¡ð­ ðð¥ð®ð ð¬ð¡ððð¢ð§ð )

Success generally increases over time since 2013 with a slight dip in 2018
Success in recent years at around 80%


# SQL

# All Launch Site Names

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_25_18 PM](https://user-images.githubusercontent.com/85125898/150127322-f168bf0a-5ac7-4771-931a-9193c8002381.png)

# First Successful Ground Landing Date

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_26_25 PM](https://user-images.githubusercontent.com/85125898/150127442-f72a6410-4783-44e9-b1c6-1118b990ad79.png)


# Launch Site Proximities Analysis 

ððð®ð§ðð¡ ðð¢ð­ð ðð¨ððð­ð¢ð¨ð§ð¬

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_28_50 PM](https://user-images.githubusercontent.com/85125898/150127741-d91824cd-733f-4fa8-a864-27584686d5b7.png)

ðð¨ð¥ð¨ð«-ðð¨ððð ððð®ð§ðð¡ ððð«ð¤ðð«ð¬

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_30_12 PM](https://user-images.githubusercontent.com/85125898/150127841-3c499431-cff9-49dd-921f-828302a6eb02.png)

ððð² ðð¨ððð­ð¢ð¨ð§ ðð«ð¨ð±ð¢ð¦ð¢ð­ð¢ðð¬

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_31_21 PM](https://user-images.githubusercontent.com/85125898/150127928-41f25ee0-b602-403f-a502-a65225127d22.png)


# DashBorad - Plotly 

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_48_55 PM](https://user-images.githubusercontent.com/85125898/150128143-c2b3b825-b795-4e98-babd-2e625d0bc554.png)

This is a preview of the Plotly dashboard. The following sides will 
show the results of EDA with visualization, 
EDA with SQL, Interactive Map with Folium, and finally the results of 
our model with about ðð% ðððð®ð«ððð².




â¢ Dashboard includes a pie chart and a scatter plot.
â¢ Pie chart can be selected to show distribution of successful landings across all 
launch sites and can be selected to show individual launch site success rates.
â¢ Scatter plot takes two inputs: All sites or individual site and payload mass 
on a slider between 0 and 10000 kg.
â¢ The pie chart is used to visualize launch site success rate.
â¢ The scatter plot can help us see 
how success varies across launch sites, payload mass, and
â¢ booster version category.


ðð®ðððð¬ð¬ðð®ð¥ ððð®ð§ðð¡ðð¬ ððð«ð¨ð¬ð¬ ððð®ð§ðð¡ ðð¢ð­ðð¬

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_33_14 PM](https://user-images.githubusercontent.com/85125898/150128335-79171364-855b-4212-bc60-f245c10d7ef6.png)

ðð¢ð ð¡ðð¬ð­ ðð®ðððð¬ð¬ ððð­ð ððð®ð§ðð¡ ðð¢ð­ð

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_34_47 PM](https://user-images.githubusercontent.com/85125898/150128435-f773dbec-5eb5-45f2-8138-54d887792b59.png)

ððð²ð¥ð¨ðð ððð¬ð¬ ð¯ð¬. ðð®ðððð¬ð¬ ð¯ð¬. ðð¨ð¨ð¬ð­ðð« ððð«ð¬ð¢ð¨ð§ ððð­ðð ð¨ð«ð²

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_35_51 PM](https://user-images.githubusercontent.com/85125898/150128499-a19addb9-76aa-439c-bf28-d9f0157b941e.png)

â¢ Plotly dashboard has a Payload range selector. However, this is set from 0-
10000 instead of the max Payload of 15600. Class indicates 1 for successful landing and 
0 for failure. Scatter plot also accounts for booster version category in color and 
number of launches in point size. In this particular range of 0-
6000, interestingly there are two failed landings with payloads of zero kg.

# Predictive Analysis ( Classification ) 

ðð¥ðð¬ð¬ð¢ðð¢ððð­ð¢ð¨ð§ ðððð®ð«ððð²

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_37_54 PM](https://user-images.githubusercontent.com/85125898/150128706-e12629d4-96df-4121-a39f-30497f897012.png)

â¢ All models had virtually the same accuracy on the test set at ðð.ðð%ðððð®ð«ððð².
It should be noted that test size is small at only sample size of 18.
â¢ This can cause large variance in accuracy 
results,such as those in Decision Tree Classifier model in repeated runs.
â¢ We likely need more data to determine the best model.


ðð¨ð§ðð®ð¬ð¢ð¨ð§ ððð­ð«ð¢ð±

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoftâ Edge 1_19_2022 5_54_20 PM](https://user-images.githubusercontent.com/85125898/150128932-75b60e32-8b87-4ce3-a41b-36b2e06aa97a.png)


# Conclusions

â¦ ðð®ð« ð­ðð¬ð¤: ð­ð¨ ððð¯ðð¥ð¨ð© ð ð¦ððð¡ð¢ð§ð ð¥ððð«ð§ð¢ð§ð  ð¦ð¨ððð¥ ðð¨ð« ðð©ððð ð ð°ð¡ð¨ ð°ðð§ð­ð¬ 
ð­ð¨ ðð¢ð ðð ðð¢ð§ð¬ð­ ðð©ðððð
â¦ ðð¡ð ð ð¨ðð¥ ð¨ð ð¦ð¨ððð¥ ð¢ð¬ ð­ð¨ ð©ð«ððð¢ðð­ ð°ð¡ðð§ ðð­ðð ð ð ð°ð¢ð¥ð¥ ð¬ð®ðððð¬ð¬ðð®ð¥ð¥ð² ð¥ðð§ð ð­ð¨ ð¬ðð¯ð ~$ððð 
ð¦ð¢ð¥ð¥ð¢ð¨ð§ ððð
â¦ ðð¬ðð ððð­ð ðð«ð¨ð¦ ð ð©ð®ðð¥ð¢ð ðð©ðððð ððð ðð§ð ð°ðð ð¬ðð«ðð©ð¢ð§ð  ðð©ðððð ðð¢ð¤ð¢ð©ððð¢ð ð©ðð ð
â¦ ðð«ððð­ðð ððð­ð ð¥ðððð¥ð¬ ðð§ð ð¬ð­ð¨ð«ðð ððð­ð ð¢ð§ð­ð¨ ð ððð ððð ððð­ðððð¬ð
â¦ ðð«ððð­ðð ð ððð¬ð¡ðð¨ðð«ð ðð¨ð« ð¯ð¢ð¬ð®ðð¥ð¢ð³ðð­ð¢ð¨ð§
â¦ ðð ðð«ððð­ðð ð ð¦ððð¡ð¢ð§ð ð¥ððð«ð§ð¢ð§ð  ð¦ð¨ððð¥ ð°ð¢ð­ð¡ ðð§ ðððð®ð«ððð² ð¨ð ðð%
â¦ ðð¥ð¥ð¨ð§ ððð¬ð¤ ð¨ð ðð©ðððð ððð§ ð®ð¬ð ð­ð¡ð¢ð¬ ð¦ð¨ððð¥ ð­ð¨ ð©ð«ððð¢ðð­ ð°ð¢ð­ð¡ ð«ðð¥ðð­ð¢ð¯ðð¥ð² ð¡ð¢ð ð¡ ðððð®ð«ððð² 
ð°ð¡ðð­ð¡ðð« ð ð¥ðð®ð§ðð¡ ð°ð¢ð¥ð¥ ð¡ðð¯ð ð ð¬ð®ðððð¬ð¬ðð®ð¥ ðð­ðð ð ð ð¥ðð§ðð¢ð§ð  ðððð¨ð«ð ð¥ðð®ð§ðð¡ ð­ð¨ ððð­ðð«ð¦ð¢ð§ð 
ð°ð¡ðð­ð¡ðð« ð­ð¡ð ð¥ðð®ð§ðð¡ ð¬ð¡ð¨ð®ð¥ð ðð ð¦ððð ð¨ð« ð§ð¨ð­
â¦ ðð ð©ð¨ð¬ð¬ð¢ðð¥ð ð¦ð¨ð«ð ððð­ð ð¬ð¡ð¨ð®ð¥ð ðð ðð¨ð¥ð¥ððð­ðð ð­ð¨ ððð­ð­ðð« ððð­ðð«ð¦ð¢ð§ð ð­ð¡ð ððð¬ð­ ð¦ððð¡ð¢ð§ð 
ð¥ððð«ð§ð¢ð§ð  ð¦ð¨ððð¥ ðð§ð ð¢ð¦ð©ð«ð¨ð¯ð ðððð®ð«ððð²
