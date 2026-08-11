# House Price Prediction Using Linear Regression

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project predicts house price from house area using Linear Regression from Scikit-learn. A sample dataset is used to train the model, and the program predicts the price for an area entered by the user.

## Features

- Creates a dataset of house areas and prices.
- Trains a Linear Regression model.
- Accepts house area as user input.
- Predicts the estimated house price.
- Prints the result in an easy-to-read format.

## Technologies

- Python
- Pandas
- Scikit-learn

## Algorithm

Linear Regression is a supervised machine learning algorithm used for continuous value prediction. In this project, it learns the relationship between house area and price.

## Dataset

| Area (sq.ft.) | Price |
| ------------: | ----: |
| 600 | 20 |
| 800 | 28 |
| 1000 | 35 |
| 1200 | 42 |
| 1400 | 50 |
| 1600 | 58 |
| 1800 | 65 |
| 2000 | 72 |
| 2200 | 80 |

## How to Run

```bash
pip install -r requirements.txt
python house-price-prediction.py
```

## Sample Output

```text
Enter House Area: 1500

Predicted House Price: 54.0 Lakhs
```

## Conclusion

This project gives a beginner-friendly example of supervised machine learning by training a Linear Regression model to estimate house price from area.
