# Metis Data Science Bootcamp | Project 2

---

## Recommendations on buying Used Vehicle Based on regression analysis

##### Web Scraping and Linear Regression Analysis

Project timeline: Two weeks

Final presentation is posted [here](https://github.com/weizhao-BME/metis-project2/blob/main/presentation/presentation_project2.pdf)

---

### Introduction

Vehicles are important tools that allow flexibly in our lives. We can go shopping any time when stores are  open regardless of weather. We can also drive from east coast to west coast of the US. My friend did so and that was an amazing trip said him. He saw many beautiful natural scenes. While our lives are facilitated by vehicles, you may want to ask yourself the questions below:

- How do I choose a car? 
- What year, make and model?
- If I have limited budget and would like  to buy a used car, what features should I focus on?

Therefore, in this project I will address these questions via a machine learning approach. First, the vehicle data and listed price will be scraped from TrueCar.com. Next, a feature analysis will be performed to determined what features of a vehicle will be used as input variables in the analysis. Finally, a linear regression model will be trained using these features with respect to the listed price, which would demonstrate the relationship between these features and the listed price. This provides a clear recommendations on how to choose a used car.  

---

### Methods

#### Data cleaning

A total of 8691 listed used cars within 25 mile from Boston MA were scraped from TrueCar.com using beautiful soup and Python. It contains make, year, mileage, engine size, city/highway MPG, fuel  type, drive type, transmission type, and engine size. A data cleaning was performed to eliminate the cars with incomplete information such as mileage, price, engine size, and city/highway MPG. In addition, the price range was limited to $8k - 40k to eliminate overly old and luxury cars. The clean data includes 6557 cars. 

#### Exploratory analysis

 The top 10 most popular vehicle makes were identified from all the scraped data to understand their market popularity. From the clean data, the range of listed vehicle price was investigated to assist for planning a budget.

#### Featuring engineering

Features used as the input of the linear regression analysis contain continuous and categorical variables. The former includes year, mileage, engine size, and city MPG. The latter includes fuel type (gas, diesel, and hybrid), drive type (AWD, FWD, and RWD), transmission type (manual and automatic), and engine type (regular or turbocharged). 

The heatmap below demonstrate pair-wise Pearson correlation coefficients between each pair of continue variables and between each continue variable and listed car price. Highway MPG is highly correlated with city MPG (correlation coefficients of 0.91). Therefore, it was discarded in the regression modeling. 



<img src="https://github.com/weizhao-BME/metis-project2/blob/main/figures/corrcoef.svg" alt="Figure 1" style="zoom:70%;" />

Three interaction features were added in order to improve the performance of linear regression model. They include 1) the interaction between drive type and city MPG as rear-wheel drive layout typically generates more power using the same amount of gas as front-wheel drive does,; 2) the interaction between transmission type and city MPG as manual transmission along with a high MPG could lead to a low price; 3) the interaction between fuel type and city MPG as a hybrid engine with a high city MPG typically induces high price. 

The figure below shows the workflow of linear regression modeling. First, the entire data was split into training and testing datasets. (80% vs 20%). Second, with the feature engineering performed above, the training dataset was used to perform linear regression with a ridge regularization for a 5-fold cross validation. Third, the best penalty strength (alpha=3.6) resulting in the minimum mean squared error was identified from a pre-defined range -0.5 - 5 (with an increment of 0.8). Next, a final linear regression model using ridge regression with the best performing penalty strength was trained based on the complete training dataset. Finally, the performance of the fitted regression model was verified using the independent testing data. 



<img src="https://github.com/weizhao-BME/metis-project2/blob/main/figures/lin_reg_workflow.png" alt="Figure 2" style="zoom: 50%;" />



---

### Results and Discussion

For all the scraped vehicles, top 10 most popular makes were listed (The figure below). Among them, the top 3 most popular vehicle makes were Toyota, Honda, and Ford, suggesting they have a hot market near the great Boston area. 



<img src="https://github.com/weizhao-BME/metis-project2/blob/main/figures/top10_makes_color_coded.svg" alt="Figure 3" style="zoom:80%;" />

The histogram below demonstrates that most of the cars of interest are priced within ~$12k - ~$30k. Therefore, it is important to plan a budge for buying a used car. This price range may not be affordable for some people to pay off at purchase. They may seek for a loan provided by dealerships. They usually offer a lower rate for used car. 



<img src="https://github.com/weizhao-BME/metis-project2/blob/main/figures/hist_listed_price_edited.png" alt="Figure 4" style="zoom:80%;" />



The subplot below on the left shows the comparison between the predicted and actual listed price. The training and testing R^2 achieved 0.74 and 0.72, respectively. The testing mean squared error is $3008, suggesting the predicted price has an average difference of $3008 relative to the actual listed price. To examine the residuals plot (middle subplot below), a linear regression analysis was performed between the residuals and predicted price. Both the resulting slope and R^2 were 0 suggesting no specific pattern and trend between the variables. The Q-Q plot (left subplot below) indicates that residuals followed a normal distribution but with left-skewness. Therefore, the listed price is more likely to be underestimated, which is consistent with the left subplot comparing the predicted and actual listed price. Underestimations were observed for the prices below $12k and over $30k as the scraped data did not contain sufficient information of this price range. However, the regression model performed well within the price range between $12k and $30k. 



![Figure 5](https://github.com/weizhao-BME/metis-project2/blob/main/figures/results_inspection.svg)



The figure below illustrates the import vehicle features that potentially affect vehicle price. Front-wheel drive layout most likely lowers car price. Other features including high mileage, high city MPG, and the use of gas also reduces price, but not as much as front-wheel drive layout. However, a very recent car equipped with a large-sized turbocharged engine and rear-wheel drive layout could have a high price. The hybrid engine also increases vehicle price. 



![Figure 6](https://github.com/weizhao-BME/metis-project2/blob/main/figures/lm_coef.svg)



---

### Conclusions

In this analysis, a exploratory investigation and linear regression were conducted in order to provide commendations on buying a used car. First, Customers may focus on the top 10 most popular makes, i.e. Toyota, Honda, and Ford. Be prepared for the price range of ~$12k - ~$30k. Finally, the following features could be focused on to choose a cheap used car:

- Front-wheel drive
- Certain amount of mileage
- Small-sized engine
- High city MPG
- Gas engine, not hybrid
- Not too recent













