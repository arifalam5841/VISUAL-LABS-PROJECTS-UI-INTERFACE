# House Price Prediction Using Linear Regression

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project predicts house price from house area using Linear Regression from Scikit-learn. A small sample dataset of areas and corresponding prices is used to train the model. After training, the model estimates the price for a house with an area of 1700 square feet.

## Features

- Creates a dataset of house areas and prices.
- Trains a Linear Regression model.
- Predicts price from area.
- Displays the predicted price in lakhs.
- Provides a simple beginner-level machine learning example.

## Technologies

- Python
- Pandas
- Scikit-learn

## Algorithm

Linear Regression is a supervised learning algorithm used for continuous value prediction. In this project, it finds the best-fit relationship between area and price.

## Dataset

| Area (sq.ft.) | Price (Lakhs) |
| ------------: | ------------: |
| 600 | 30 |
| 800 | 40 |
| 1000 | 50 |
| 1200 | 60 |
| 1500 | 75 |
| 1800 | 90 |
| 2000 | 100 |
| 2200 | 110 |
| 2500 | 125 |

## How to Run

```bash
pip install pandas scikit-learn
python house_price_prediction.py
```

## Sample Output

```text
Predicted House Price : 85.0 Lakhs
```

## Conclusion

This project shows how Linear Regression can estimate house prices based on area. It gives a simple introduction to supervised learning by training on sample data and predicting the value for a new area.
