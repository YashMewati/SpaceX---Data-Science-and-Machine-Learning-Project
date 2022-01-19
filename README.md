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

# Success Rate vs. Orbit Type

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_21_27 PM](https://user-images.githubusercontent.com/85125898/150127010-de5c1fdb-876d-483e-a965-0bfc71749785.png)

• Success Rate Scale with 0 as 0%
• 0.6 as 60% 1 as 100%
• ES-L1 (1), GEO(1), HEO (1) have 100% success rate (sample sizesin parenthesis) SSO(5) has 100% success rate
• VLEO (14) has decentsuccess rate and attempts
• SO (1) has 0% success rate
• GTO (27) has the around 50% success rate butlargestsampl

# Launch Success Yearly Trend

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_23_48 PM](https://user-images.githubusercontent.com/85125898/150127115-643d76af-3c4a-4b2f-8b22-b4114f2d5db5.png)

• 𝟗𝟓% 𝐜𝐨𝐧𝐟𝐢𝐝𝐞𝐧𝐜𝐞 𝐢𝐧𝐭𝐞𝐫𝐯𝐚𝐥 (𝐥𝐢𝐠𝐡𝐭 𝐛𝐥𝐮𝐞 𝐬𝐡𝐚𝐝𝐢𝐧𝐠)

Success generally increases over time since 2013 with a slight dip in 2018
Success in recent years at around 80%


# SQL

# All Launch Site Names

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_25_18 PM](https://user-images.githubusercontent.com/85125898/150127322-f168bf0a-5ac7-4771-931a-9193c8002381.png)

# First Successful Ground Landing Date

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_26_25 PM](https://user-images.githubusercontent.com/85125898/150127442-f72a6410-4783-44e9-b1c6-1118b990ad79.png)


# Launch Site Proximities Analysis 

𝐋𝐚𝐮𝐧𝐜𝐡 𝐒𝐢𝐭𝐞 𝐋𝐨𝐜𝐚𝐭𝐢𝐨𝐧𝐬

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_28_50 PM](https://user-images.githubusercontent.com/85125898/150127741-d91824cd-733f-4fa8-a864-27584686d5b7.png)

𝐂𝐨𝐥𝐨𝐫-𝐂𝐨𝐝𝐞𝐝 𝐋𝐚𝐮𝐧𝐜𝐡 𝐌𝐚𝐫𝐤𝐞𝐫𝐬

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_30_12 PM](https://user-images.githubusercontent.com/85125898/150127841-3c499431-cff9-49dd-921f-828302a6eb02.png)

𝐊𝐞𝐲 𝐋𝐨𝐜𝐚𝐭𝐢𝐨𝐧 𝐏𝐫𝐨𝐱𝐢𝐦𝐢𝐭𝐢𝐞𝐬

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_31_21 PM](https://user-images.githubusercontent.com/85125898/150127928-41f25ee0-b602-403f-a502-a65225127d22.png)


# DashBorad - Plotly 

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_48_55 PM](https://user-images.githubusercontent.com/85125898/150128143-c2b3b825-b795-4e98-babd-2e625d0bc554.png)

This is a preview of the Plotly dashboard. The following sides will 
show the results of EDA with visualization, 
EDA with SQL, Interactive Map with Folium, and finally the results of 
our model with about 𝟖𝟑% 𝐚𝐜𝐜𝐮𝐫𝐚𝐜𝐲.




• Dashboard includes a pie chart and a scatter plot.
• Pie chart can be selected to show distribution of successful landings across all 
launch sites and can be selected to show individual launch site success rates.
• Scatter plot takes two inputs: All sites or individual site and payload mass 
on a slider between 0 and 10000 kg.
• The pie chart is used to visualize launch site success rate.
• The scatter plot can help us see 
how success varies across launch sites, payload mass, and
• booster version category.


𝐒𝐮𝐜𝐜𝐞𝐬𝐬𝐟𝐮𝐥 𝐋𝐚𝐮𝐧𝐜𝐡𝐞𝐬 𝐀𝐜𝐫𝐨𝐬𝐬 𝐋𝐚𝐮𝐧𝐜𝐡 𝐒𝐢𝐭𝐞𝐬

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_33_14 PM](https://user-images.githubusercontent.com/85125898/150128335-79171364-855b-4212-bc60-f245c10d7ef6.png)

𝐇𝐢𝐠𝐡𝐞𝐬𝐭 𝐒𝐮𝐜𝐜𝐞𝐬𝐬 𝐑𝐚𝐭𝐞 𝐋𝐚𝐮𝐧𝐜𝐡 𝐒𝐢𝐭𝐞

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_34_47 PM](https://user-images.githubusercontent.com/85125898/150128435-f773dbec-5eb5-45f2-8138-54d887792b59.png)

𝐏𝐚𝐲𝐥𝐨𝐚𝐝 𝐌𝐚𝐬𝐬 𝐯𝐬. 𝐒𝐮𝐜𝐜𝐞𝐬𝐬 𝐯𝐬. 𝐁𝐨𝐨𝐬𝐭𝐞𝐫 𝐕𝐞𝐫𝐬𝐢𝐨𝐧 𝐂𝐚𝐭𝐞𝐠𝐨𝐫𝐲

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_35_51 PM](https://user-images.githubusercontent.com/85125898/150128499-a19addb9-76aa-439c-bf28-d9f0157b941e.png)

• Plotly dashboard has a Payload range selector. However, this is set from 0-
10000 instead of the max Payload of 15600. Class indicates 1 for successful landing and 
0 for failure. Scatter plot also accounts for booster version category in color and 
number of launches in point size. In this particular range of 0-
6000, interestingly there are two failed landings with payloads of zero kg.

# Predictive Analysis ( Classification ) 

𝐂𝐥𝐚𝐬𝐬𝐢𝐟𝐢𝐜𝐚𝐭𝐢𝐨𝐧 𝐀𝐜𝐜𝐮𝐫𝐚𝐜𝐲

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_37_54 PM](https://user-images.githubusercontent.com/85125898/150128706-e12629d4-96df-4121-a39f-30497f897012.png)

• All models had virtually the same accuracy on the test set at 𝟖𝟑.𝟑𝟑%𝐚𝐜𝐜𝐮𝐫𝐚𝐜𝐲.
It should be noted that test size is small at only sample size of 18.
• This can cause large variance in accuracy 
results,such as those in Decision Tree Classifier model in repeated runs.
• We likely need more data to determine the best model.


𝐂𝐨𝐧𝐟𝐮𝐬𝐢𝐨𝐧 𝐌𝐚𝐭𝐫𝐢𝐱

![Spacex-DS-ML Project - Yash Mewati pdf - Personal - Microsoft​ Edge 1_19_2022 5_54_20 PM](https://user-images.githubusercontent.com/85125898/150128932-75b60e32-8b87-4ce3-a41b-36b2e06aa97a.png)


# Conclusions

◦ 𝐎𝐮𝐫 𝐭𝐚𝐬𝐤: 𝐭𝐨 𝐝𝐞𝐯𝐞𝐥𝐨𝐩 𝐚 𝐦𝐚𝐜𝐡𝐢𝐧𝐞 𝐥𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐦𝐨𝐝𝐞𝐥 𝐟𝐨𝐫 𝐒𝐩𝐚𝐜𝐞 𝐘 𝐰𝐡𝐨 𝐰𝐚𝐧𝐭𝐬 
𝐭𝐨 𝐛𝐢𝐝 𝐚𝐠𝐚𝐢𝐧𝐬𝐭 𝐒𝐩𝐚𝐜𝐞𝐗
◦ 𝐓𝐡𝐞 𝐠𝐨𝐚𝐥 𝐨𝐟 𝐦𝐨𝐝𝐞𝐥 𝐢𝐬 𝐭𝐨 𝐩𝐫𝐞𝐝𝐢𝐜𝐭 𝐰𝐡𝐞𝐧 𝐒𝐭𝐚𝐠𝐞 𝟏 𝐰𝐢𝐥𝐥 𝐬𝐮𝐜𝐜𝐞𝐬𝐬𝐟𝐮𝐥𝐥𝐲 𝐥𝐚𝐧𝐝 𝐭𝐨 𝐬𝐚𝐯𝐞 ~$𝟏𝟎𝟎 
𝐦𝐢𝐥𝐥𝐢𝐨𝐧 𝐔𝐒𝐃
◦ 𝐔𝐬𝐞𝐝 𝐝𝐚𝐭𝐚 𝐟𝐫𝐨𝐦 𝐚 𝐩𝐮𝐛𝐥𝐢𝐜 𝐒𝐩𝐚𝐜𝐞𝐗 𝐀𝐏𝐈 𝐚𝐧𝐝 𝐰𝐞𝐛 𝐬𝐜𝐫𝐚𝐩𝐢𝐧𝐠 𝐒𝐩𝐚𝐜𝐞𝐗 𝐖𝐢𝐤𝐢𝐩𝐞𝐝𝐢𝐚 𝐩𝐚𝐠𝐞
◦ 𝐂𝐫𝐞𝐚𝐭𝐞𝐝 𝐝𝐚𝐭𝐚 𝐥𝐚𝐛𝐞𝐥𝐬 𝐚𝐧𝐝 𝐬𝐭𝐨𝐫𝐞𝐝 𝐝𝐚𝐭𝐚 𝐢𝐧𝐭𝐨 𝐚 𝐃𝐁𝟐 𝐒𝐐𝐋 𝐝𝐚𝐭𝐚𝐛𝐚𝐬𝐞
◦ 𝐂𝐫𝐞𝐚𝐭𝐞𝐝 𝐚 𝐝𝐚𝐬𝐡𝐛𝐨𝐚𝐫𝐝 𝐟𝐨𝐫 𝐯𝐢𝐬𝐮𝐚𝐥𝐢𝐳𝐚𝐭𝐢𝐨𝐧
◦ 𝐖𝐞 𝐜𝐫𝐞𝐚𝐭𝐞𝐝 𝐚 𝐦𝐚𝐜𝐡𝐢𝐧𝐞 𝐥𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐦𝐨𝐝𝐞𝐥 𝐰𝐢𝐭𝐡 𝐚𝐧 𝐚𝐜𝐜𝐮𝐫𝐚𝐜𝐲 𝐨𝐟 𝟖𝟑%
◦ 𝐀𝐥𝐥𝐨𝐧 𝐌𝐚𝐬𝐤 𝐨𝐟 𝐒𝐩𝐚𝐜𝐞𝐘 𝐜𝐚𝐧 𝐮𝐬𝐞 𝐭𝐡𝐢𝐬 𝐦𝐨𝐝𝐞𝐥 𝐭𝐨 𝐩𝐫𝐞𝐝𝐢𝐜𝐭 𝐰𝐢𝐭𝐡 𝐫𝐞𝐥𝐚𝐭𝐢𝐯𝐞𝐥𝐲 𝐡𝐢𝐠𝐡 𝐚𝐜𝐜𝐮𝐫𝐚𝐜𝐲 
𝐰𝐡𝐞𝐭𝐡𝐞𝐫 𝐚 𝐥𝐚𝐮𝐧𝐜𝐡 𝐰𝐢𝐥𝐥 𝐡𝐚𝐯𝐞 𝐚 𝐬𝐮𝐜𝐜𝐞𝐬𝐬𝐟𝐮𝐥 𝐒𝐭𝐚𝐠𝐞 𝟏 𝐥𝐚𝐧𝐝𝐢𝐧𝐠 𝐛𝐞𝐟𝐨𝐫𝐞 𝐥𝐚𝐮𝐧𝐜𝐡 𝐭𝐨 𝐝𝐞𝐭𝐞𝐫𝐦𝐢𝐧𝐞 
𝐰𝐡𝐞𝐭𝐡𝐞𝐫 𝐭𝐡𝐞 𝐥𝐚𝐮𝐧𝐜𝐡 𝐬𝐡𝐨𝐮𝐥𝐝 𝐛𝐞 𝐦𝐚𝐝𝐞 𝐨𝐫 𝐧𝐨𝐭
◦ 𝐈𝐟 𝐩𝐨𝐬𝐬𝐢𝐛𝐥𝐞 𝐦𝐨𝐫𝐞 𝐝𝐚𝐭𝐚 𝐬𝐡𝐨𝐮𝐥𝐝 𝐛𝐞 𝐜𝐨𝐥𝐥𝐞𝐜𝐭𝐞𝐝 𝐭𝐨 𝐛𝐞𝐭𝐭𝐞𝐫 𝐝𝐞𝐭𝐞𝐫𝐦𝐢𝐧𝐞 𝐭𝐡𝐞 𝐛𝐞𝐬𝐭 𝐦𝐚𝐜𝐡𝐢𝐧𝐞 
𝐥𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐦𝐨𝐝𝐞𝐥 𝐚𝐧𝐝 𝐢𝐦𝐩𝐫𝐨𝐯𝐞 𝐚𝐜𝐜𝐮𝐫𝐚𝐜𝐲
