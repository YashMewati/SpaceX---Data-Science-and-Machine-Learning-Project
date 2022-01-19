# SPACEX - DATA SCIENCE AND MACHINE LEARNING PROJECT

𝐈𝐧𝐭𝐫𝐨𝐝𝐮𝐜𝐭𝐢𝐨𝐧

• 𝐂𝐨𝐥𝐥𝐞𝐜𝐭𝐞𝐝 𝐝𝐚𝐭𝐚 𝐟𝐫𝐨𝐦 𝐩𝐮𝐛𝐥𝐢𝐜 𝐒𝐩𝐚𝐜𝐞𝐗 𝐀𝐏𝐈 
𝐚𝐧𝐝 𝐒𝐩𝐚𝐜𝐞𝐗 𝐖𝐢𝐤𝐢𝐩𝐞𝐝𝐢𝐚 𝐩𝐚𝐠𝐞.Created labels column ‘class’ which 
classifies successful landings. Explored data using SQL, visualization, folium maps, a
nd dashboards. Gathered relevant columns to be used as features. Changed all cat
egorical variables to binary using one 
hot encoding. Standardized data and used GridSearchCV to find best parameters for machine learning models. 
Visualize accuracy score of all models.

• 𝐅𝐨𝐮𝐫 𝐦𝐚𝐜𝐡𝐢𝐧𝐞 𝐥𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐦𝐨𝐝𝐞𝐥𝐬 𝐰𝐞𝐫𝐞 𝐩𝐫𝐨𝐝𝐮𝐜𝐞𝐝:
Logistic Regression, Support Vector Machine, Decision Tree Classifier, and K Nearest 
Neighbors. All produced similar results with 𝐚𝐜𝐜𝐮𝐫𝐚𝐜𝐲 𝐫𝐚𝐭𝐞 𝐨𝐟 𝐚𝐛𝐨𝐮𝐭 𝟖𝟑.𝟑𝟑%. All 
models over predicted successful landings. More data is needed for better model determination and accuracy.

# METHODOLOGY

• 𝐃𝐚𝐭𝐚 𝐜𝐨𝐥𝐥𝐞𝐜𝐭𝐢𝐨𝐧 𝐦𝐞𝐭𝐡𝐨𝐝𝐨𝐥𝐨𝐠𝐲:

• Combined data from SpaceX public API and SpaceX 
Wikipedia page

• 𝐏𝐞𝐫𝐟𝐨𝐫𝐦 𝐝𝐚𝐭𝐚 𝐰𝐫𝐚𝐧𝐠𝐥𝐢𝐧𝐠

• Classifying true 
landings as successful and unsuccessful otherwise
• Perform exploratory data analysis(EDA) using visualization and SQL
• Perform interactive visual analytics using 𝐅𝐨𝐥𝐢𝐮𝐦 and 𝐏𝐥𝐨𝐭𝐥𝐲 𝐃𝐚𝐬𝐡
• Perform predictive analysis using classificationmodels
• Tuned models using 𝐆𝐫𝐢𝐝𝐒𝐞𝐚𝐫𝐜𝐡𝐂𝐕

# Data Collection

• Data collection process involved a combination of API requests from Space X public API 
and web scraping data from a table in Space X’s Wikipedia entry.
• The next slide will show the flowchart of data collection from API and 
the one after will show the flowchart of data collection from webscraping.
• Space X API Data Columns:
• FlightNumber, Date, BoosterVersion, PayloadMass, Orbit, 
LaunchSite, Outcome, Flights, GridFins,
• Reused, Legs, LandingPad, Block, ReusedCount, Serial, Longitude, Latitude
• Wikipedia Webscrape Data Columns:
• Flight No., Launch site, Payload, PayloadMass, Orbit, Customer, Launch outcome,
Version Booster, Booster landing, Date, Time


# Data Collection – SpaceX API

https://github.com/YashMewati/SpaceX---Data-Science-and-Machine-Learning-Project/blob/main/Week1%20_%20SpaceX%20Falcon%20Data%20Collection-Wrangling.ipynb

# Data Collection - Scraping

https://github.com/YashMewati/SpaceX---Data-Science-and-Machine-Learning-Project/blob/main/Web%20scraping%20Falcon%209%20and%20Falcon%20Heavy%20Launches%20Records%20from%20Wikipedia.ipynb

# EDA 

• Exploratory Data Analysis performed on 
variables Flight Number, Payload Mass, Launch Site, Orbit, Class and Year.

• 𝐏𝐥𝐨𝐭𝐬 𝐔𝐬𝐞𝐝:

• Flight Number vs. Payload Mass, Flight Number vs. Launch Site, Payload Mass
vs. Launch Site, Orbit vs. Success Rate, Flight Number vs. Orbit, Payload vs Orb
it, and Success Yearly Trend
• Scatter plots, line charts, and bar plots were used to compare relationships 
between variables to
• decide if a relationship exists so that they could be used in training the 
machine learning model

# Flight Number vs. Launch Site

• 𝐆𝐫𝐞𝐞𝐧 𝐢𝐧𝐝𝐢𝐜𝐚𝐭𝐞𝐬 𝐬𝐮𝐜𝐜𝐞𝐬𝐬𝐟𝐮𝐥 𝐥𝐚𝐮𝐧𝐜𝐡; 𝐏𝐮𝐫𝐩𝐥𝐞 𝐢𝐧𝐝𝐢𝐜𝐚𝐭𝐞𝐬 𝐮𝐧𝐬𝐮𝐜𝐜𝐞𝐬𝐬𝐟𝐮𝐥 𝐥𝐚𝐮𝐧𝐜𝐡.

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_16_18 PM](https://user-images.githubusercontent.com/85125898/150124039-f94eca1a-3335-431a-ba19-c519e1b8fb5f.png)

# Payload vs. Launch Site

• 𝐆𝐫𝐞𝐞𝐧 𝐢𝐧𝐝𝐢𝐜𝐚𝐭𝐞𝐬 𝐬𝐮𝐜𝐜𝐞𝐬𝐬𝐟𝐮𝐥 𝐥𝐚𝐮𝐧𝐜𝐡; 𝐏𝐮𝐫𝐩𝐥𝐞 𝐢𝐧𝐝𝐢𝐜𝐚𝐭𝐞𝐬 𝐮𝐧𝐬𝐮𝐜𝐜𝐞𝐬𝐬𝐟𝐮𝐥 𝐥𝐚𝐮𝐧𝐜𝐡.

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_19_44 PM](https://user-images.githubusercontent.com/85125898/150124344-c2ee0d8c-877f-46e8-a03b-b2362cc3d0c8.png)



