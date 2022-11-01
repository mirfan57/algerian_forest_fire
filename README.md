
# Algerian Forest Fire Prediction
<!--**Heroku App** 
(https://forestfire-predictions.herokuapp.com) 
https://user-images.githubusercontent.com/86904142/185993973-57c8f915-c1a6-4251-bd4c-9275dee94235.png-->
**User Interface**

![front_ui](https://user-images.githubusercontent.com/103884216/199303258-77206bdd-6c3f-4861-98bb-c9decc93da99.jpg)

## [Heroku App Link](https://algerian-forest-fire-app.herokuapp.com/)

# Demo

https://user-images.githubusercontent.com/103884216/199303070-307774c9-bdfe-446f-823f-4fc2f147daf1.mp4


## A brief explanation of the project's objectives.

Forest Fire Prediction is a Supervised Machine learning problem statements.
Using Regression and Classification Algorithm, Regression and Classification Model is build that detected future fires based on certain Weather report.

## Libraries Implemented 

**Data Pre-Processing**

- Numpy, Pandas, Matplotlib, Seaborn

**Feature Selection**

- Variance Inflation Factor

**Model Building**

- Sklearn, statsmodels

**Hyperparameter Tuning**

- Randomized SearchCV, Grid SearchCV

**Model Selection**

- Repeated Stratified KFold

## About the Dataset

**Algerian Forest Fires Dataset**

I used a UCI dataset on forest fires in Algeria. The **Bejaia and Sidi Bel-abbes** areas of Algeria are represented by the dataset's observations and records on forest fires are loaded. This dataset's time period runs from **June 2012 to September 2012**. In this case, we investigated if a few machine learning algorithms might accurately predict forest fires in certain locations using specific weather information.

**Dataset taken from:** [Link](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset++#)

***Data Set Information:***

- The dataset contains 244 occurrences that aggregate data from two locations of Algeria: the **Sidi Bel-abbes** region in northwest Algeria and the **Bejaia** region in northeast Algeria.
- 122 instances for each region.

- The data collected in a span of June 2012 to September 2012.
- The dataset includes a total of 12 attributes including 11 feature attributes and 1 output attribute depending on the task performed. 
- The 244 instances have been classified into **fire** (138 classes) and **not fire** (106 classes) classes.

**Attribute Information:**

**1. Date :** (DD/MM/YYYY) Day, month ('june' to 'september'), year (2012)

**Weather data observations:**

**2. Temperature :** Maximum day temperature in degrees Celsius: 22 to 42

**3. RH :** Relative Humidity in %: 21 to 90

**4. Ws :** Wind speed in km/h: 6 to 29

**5. Rain:** Entire day in mm: 0 to 16.8

**Fire Weather Index Components:**

**6. Fine Fuel Moisture Code (FFMC)** index from the FWI system: 28.6 to 92.5

**7. Duff Moisture Code (DMC)** index from the FWI system: 1.1 to 65.9

**8. Drought Code (DC)** index from the FWI system: 7 to 220.4

**9. Initial Spread Index (ISI)** index from the FWI system: 0 to 18.5

**10. Buildup Index (BUI)** index from the FWI system: 1.1 to 68

**11. Fire Weather Index (FWI)** Index: 0 to 31.1

**12. Classes:** two classes, namely **Fire** and **not Fire**

# Quick Walkthrough

1. Loading Data
2. Data Pre-Processing
3. Exploratory Data Analysis(EDA)
4. Feature Engineering
5. Feature Selection
6. Model Building
7. Model Selection
8. Hyperparameter Tuning
9. Performence Metrics
10. Flask web framework

## Model Building

### Regression 

- For regression analysis **Temperature** is considered as dependent feature.

**Models Trained:** 

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic NET
5. Support Vector Regressor
6. K-Nearest Neighbour Regressor
7. Decision Tree Regressor
8. Random Forest Regressor


### Classification

- For Classification purpose, **Classes** which is a binary **(fire, not fire)** categorical column is considered as the target column.

**Models Trained:** 

1. Logistic Regression
2. K Neighbors Classifier
3. Decision Tree Classifier
4. Gaussian NB Classifier
5. Random Forest Classifier
6. Support Vector Classifier
7. XGBoost Classifier


## Model Selection

Hyper-parameter tuning performed using **RandomizedsearchCV** and **GridSearchCV** for the best performing model in case of both Regression and Classification tasks.

- One metric used to evaluate the performance of regression-based machine learning models is the R2 score. We use the **R2 score** and **adjusted R2 score** metrics to choose the optimal model for regression.

- In classification, Decision Tree model is tuned with **cost complexity pruning**, which gives the effective alphas of subtrees during pruning. We fit the decision tree model with these ccp_alpha values in order to achieve a testing score close to the training score.

- The **F1 score** is the initial criteria for picking models for hyperparameter tuning in case of classification, however since achieving 100% accuracy signals overfitting problems, we choose the **Repeated Stratified Kfold Cross-Validation** metrics later on. It repeats the **Stratified KFold** n times with different randomization in each repetition and evaluates the score by cross-validation.
- The best average cross-validation accuracy achieved by the model is used for prediction purposes.


### WebApp
* The web framework employed for this model deployment is Flask.

### Deployment steps in Heroku 

* Create new repo in Github and push all the data using `Git`.

* Install Heroku CLI and then login using `heroku login`. 
* Create an app on Heroku.
* Move to the project folder and connect with your app using `heroku git:remote -a <your_app_name>`
* Push your code with `git push heroku master`
* You have successfully deployed in Heroku. 

### Code
* **Algerian forest fire_cleaned.csv** contains the dataset after inital data pre-processing and cleaning step.
* For codes concerning EDA, please look for **Algerian Forest Fire EDA.ipynb** file. Whereas ML Algorithms can be found in **Algerian forest fire regression.ipynb** and **Algerian Forest fire classification.ipynb** files.

### **Technologies used**
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)


### **Tools used**
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Postman](https://img.shields.io/badge/Postman-eeeeee?style=for-the-badge&logo=postman&logoColor=FF6C37&labelColor=fefefe)



<!-- CONTACT -->
## Contact
[![Mohammad Irfan | LinkedIn](https://img.shields.io/badge/Mohammad_Irfan-eeeeee?style=for-the-badge&logo=linkedin&logoColor=ffffff&labelColor=0A66C2)][reach_linkedin]
[![mohdirfan57 | G Mail](https://img.shields.io/badge/mohdirfan57-eeeeee?style=for-the-badge&logo=gmail&logoColor=ffffff&labelColor=EA4335)][reach_gmail]

[reach_linkedin]: https://www.linkedin.com/in/mirfan57/
[reach_gmail]: mailto:mohdirfan57@gmail.com?subject=Github


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/mirfan57)
<!--
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ashishkumar-rana/) -->
