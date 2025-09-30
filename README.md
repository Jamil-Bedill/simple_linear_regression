# Simple Linear Regression: Marketing Sales Analysis
 ## Project Overview
This project demonstrates how to apply simple linear regression to a fictional marketing dataset (marketing_sales_data.csv). The dataset records company spending on various marketing channels, including TV, radio, social media, and influencer, along with the revenue generated from these campaigns in millions.
The goal of this analysis is to understand the relationship between marketing investments and sales revenue. Specifically, we explore how radio promotion budgets impact company sales.
## Dataset
- **File**:  marketing_sales_data.csv -> you can find this dataset in the repository
- **Variables**
  -TV (categorical: Low, Medium, High)
  
  - TV (categorical: Low, Medium, High)
  - Radio (numeric: promotion budget in millions)
  - Social Media (numeric: promotion budget in millions)
  - Influencer (categorical: Nano, Micro, Macro, Mega)
  - Sales (numeric: revenue in millions)
## Steps in the Analysis 
### 1. Data Importing and Cleaning
- Imported necessary Python libraries: numpy, pandas, matplotlib, seaborn, statsmodels.
   ```
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   import statsmodels.api as sm
   from statsmodels.formula.api import ols
   ```
- Loaded the dataset into a DataFrame.
  ```
  data = pd.read_csv('/content/marketing_sales_data.csv')
  # Now we show 10 rows of the dataset to explore the dataset.
  data.head(10)
  ```
  <img width="483" height="350" alt="image" src="https://github.com/user-attachments/assets/0b41d04f-167b-4540-838f-39bb0f1e7399" />

- Detected and removed rows with missing values (3 rows in total).
  ```
  data.isna().sum()
  ```
  <img width="163" height="209" alt="image" src="https://github.com/user-attachments/assets/cfe4a1c1-e50e-42b7-8e5f-b3e9ba4ff040" />
  
  ```
  data.isnull().any(axis =1).sum()
  ```
  <img width="115" height="28" alt="image" src="https://github.com/user-attachments/assets/04e42e89-3ff3-4c37-a6a6-291c08dcbe71" />
  
Three rows have missing values. Next, let's drop the rows that contain missing values. Data cleaning makes data more usable for analysis and regression. Then, we verify that the resulting data does not contain any rows with missing values.
  
  ```
  data.dropna(axis = 0, inplace = True)
  # Verifying if the missing values have been removed 
  data.isnull().any(axis = 1).sum()
  ```
  <img width="113" height="31" alt="image" src="https://github.com/user-attachments/assets/8bedc9f8-a109-4035-8ad5-ffbcf7786cc9" />

- Verified dataset integrity after cleaning.
### 2. Exploratory Data Analysis 
To begin, we will create pairwise plots of the variables in the dataset. These plots will help us:
 - Visualise relationships between variables
 - Detect possible linear patterns
 - Spot unusual values or potential outliers

This step will give us an initial understanding of whether linear regression is an appropriate choice for analysing the data.
 ```
sns.pairplot(data)
```
<img width="750" height="737" alt="image" src="https://github.com/user-attachments/assets/0afa42e4-1eac-4d32-b17d-70129b31a56e" />

If we look at the pairplot, we can see that the relation between Radio promotional budget and Sales is strongly linear. We can see that we can conduct a simple linear regression analysis since we are exploring the relationship of a continuous dependent variable(Sales) and one independent variable. 
### 3. Model Building
Lets select the columns needed for the model
 ```
ols_data = data[['Sales','Radio']]
ols_data.reset_index(inplace = True, drop = True)
ols_data.head() # confirming the result
```
<img width="212" height="201" alt="image" src="https://github.com/user-attachments/assets/0055cc63-9e9c-4832-b850-a11c9f4d259b" />
Now, let's build the model.
```
# First we write OLS formula 
ols_formula = 'Sales ~ Radio'
# We implement the ordinary least squares (OLS) approach for linear regression.
OLS = ols(formula = ols_formula, data = ols_data)
# Now, create a linear regression model for the data and fit the model to the data.
model = OLS.fit()
```
### 4 Model Result and Evaluation
We begin by getting a summary of the results from the model.
 ```
model.summary()
```
<img width="798" height="442" alt="image" src="https://github.com/user-attachments/assets/878abf06-9b14-4bb4-9168-c51635e6edaa" />

The ordinary least squares (OLS) regression analysis shows a strong positive relationship between company sales and radio advertising budgets. The model explains 75.7% of the variation in sales (R-squared = 0.757) and is highly significant (F-statistic = 1768, p < 0.001). The estimated regression equation is:

  Sales = 41.53 + 8.17 × Radio

This means that, on average, an additional million dollars spent on radio advertising is associated with an increase of about 8.17 million dollars in sales. The slope coefficient is statistically significant (p < 0.001), with a 95% confidence interval ranging from 7.79 to 8.56, indicating strong evidence for the effect. Diagnostic tests suggest that the model assumptions are satisfied: the residuals are approximately normally distributed (Omnibus p = 0.322, Jarque-Bera p = 0.329) and show no strong autocorrelation (Durbin-Watson ≈ 1.88). Overall, the results indicate that radio advertising has a substantial and reliable impact on sales.
### 5. Model Assumptions
1. Linearity Assumption: The linearity assumption says that the relationship between the independent variable and the dependent variable is linear, a straight line. The errors are simply random fluctuations around this line.
 ```
#  Let's create a regplot from seaborn
sns.regplot(data = ols_data, x = 'Radio', y = 'Sales', color = 'darkred')
```
<img width="611" height="460" alt="image" src="https://github.com/user-attachments/assets/4d41749c-1b59-4212-b2fc-0213abb945f1" />

The graph shows that the linearity assumption is met as the data points are approximately linearly connected along the best-fit line. Now, let's check the normality assumption. Get the residuals from the model.

2. Independence Assumption: We assume that the observations are independent. 
3. Normality Assumption: 	The errors (residuals) are normally distributed (bell-shaped curve) with a mean of zero. This is crucial for valid hypothesis testing and confidence intervals, especially with small samples. We can check this assumption by plotting the residuals of the model or by a Q-Q plot. 
 ```
#  Get the residuals from the model.
residuals = model.resid
#Let's plot the residuals and see if they make a normal distribution.
fig, axes = plt.subplots(1,2, figsize = (8,4))
sns.histplot(residuals, kde = True, ax = axes[0])
axes[0]. set_title('Distribution of Model Residuals')
sm.qqplot(residuals, line = 's', fit = True , ax = axes[1])
axes[1].set_title('Q-Q Plot of Model Residuals')
plt.show()
```
<img width="723" height="401" alt="image" src="https://github.com/user-attachments/assets/f81f5b49-e81b-4f27-8be8-83fca2d73416" />
The histogram of residuals shows that they are normally distributed, and the Q-Q plot shows that the datapoints are closely following a straight line; both of these graphs confirm that the assumption of normality is met. 
4. Equal Variance (Homoscedasticity): The variance (spread) of the errors is constant across all levels of the independent variables. 
We can check this assumption by plotting a scatterplot of fitted values against the model residuals. 
 ```
# Create a scatterplot of residuals against fitted values.
# Get fitted values.
fitted_values = model.fittedvalues
# Create a scatterplot of residuals against fitted values.
sns.scatterplot(x = fitted_values, y = residuals)
plt.title('Scatter Plot of Residuals and Fitted Values')
plt.axhline(0, linestyle = '--', color = 'red')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()
```
<img width="639" height="459" alt="image" src="https://github.com/user-attachments/assets/2c97c1d4-1187-4190-8802-0ac4e9615917" />

The scatterplot shows that the points are randomly distributed, creating and cloud shape, and do not follow a specific pattern, indicating that the homoscedasticity is met.
## Conclusion
We conducted a simple linear regression to examine the relationship between company sales and radio advertising budgets. The analysis shows a strong positive association, with an intercept of 41.53 and a slope of 8.17. This indicates that, on average, each additional million dollars spent on radio advertising is associated with an increase of approximately 8.17 million dollars in sales. The model explains 75.7% of the variation in sales (R-squared = 0.757), and the effect of radio advertising is highly statistically significant (p-value < 0.001). The 95% confidence interval for the slope ranges from 7.79 to 8.56, suggesting a high level of certainty in the estimated effect. Residual diagnostics indicate that the model assumptions are reasonably met, with residuals approximately normally distributed (Omnibus p = 0.322, Jarque-Bera p = 0.329) and no serious autocorrelation (Durbin-Watson ≈ 1.88). Overall, these results provide strong evidence that investing in radio advertising is likely to increase sales.
## Recommendation
The regression results provide strong evidence that radio advertising has a significant and positive impact on company sales. Given that each additional million dollars spent on radio advertising is associated with an average increase of 8.17 million dollars in sales, companies should consider increasing their investment in radio promotion as part of their marketing strategy. However, it is important to note that this analysis is based on one dataset and a single predictor variable. To make more informed decisions, companies are encouraged to explore additional factors such as television, digital, or print advertising, as well as market conditions, to develop a more comprehensive understanding of what drives sales performance.

