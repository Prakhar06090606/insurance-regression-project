# insurance-regression-project

Project Description
This project develops a Supervised Machine Learning model to predict individual medical insurance costs based on personal attributes such as age, BMI, and smoking status. Using the Medical Cost Personal Dataset, the project implements a standard Linear Regression model as a baseline and compares it against an advanced linear Regression model optimized with Stochastic Gradient Descent (SGD). The goal is to accurately forecast continuous financial values , demonstrating the application of regression techniques in real world healthcare finance.

1. Exploratory Data Analysis (EDA)
Before modeling, we analyze the data to understand underlying patterns.

Distribution Analysis (Graph 1): We visualize the target variable (charges) to check for skewness. Most insurance charges are low, but a "long tail" of expensive cases exists.

Feature Analysis (Graph 2): We use boxplots to observe the relationship between categorical features and costs. This confirms that smokers have significantly higher median charges than non smokers, identifying it as a critical predictive feature.

Correlation Matrix (Graph 3): We calculate Pearson correlation coefficients to quantify linear relationships.
The Math: It measures the strength of the linear association between two variables, ranging from -1 to +1. A value close to 1 implies a strong positive relationship.

3. Data Preprocessing

One-Hot Encoding:
Why: Regression algorithms require numerical input. We convert categorical text data (e.g., "southwest", "female") into binary vectors (0s and 1s).
The Math: A categorical variable x with k categories is transformed into k-1 binary variables to avoid multicollinearity.

Train-Test Split:
Why: We split the data into a Training Set (80%) to teach the model and a Test Set (20%) for final evaluation.
The Math: This ensures the model's accuracy is tested on unseen data, preventing overfitting (where the model memorizes the training data instead of learning general rules).

3. Model 1: Simple Linear Regression
Why: This serves as our baseline model to establish a minimum performance standard.
The Math: The model attempts to fit a straight line through the data by minimizing the error between predicted and actual values. It solves the equation: Y= b0 + b1X1 + b2X2 + bnXn
Where $Y$ is the predicted charge, X are the features (age, bmi), and b are the weights learned by the Ordinary Least Squares (OLS) method.

4. Evaluation Metrics
We evaluate performance using two standard metrics for regression:

R-Squared (R^2): Represents the percentage of the variance in medical costs that our model can explain. A higher score (closer to 1.0) is better.

RMSE (Root Mean Squared Error): The average distance between our predictions and the actual values, measured in dollars. Lower is better.

5. Model 2: Polynomial Regression with Gradient Descent
This advanced pipeline improves accuracy by addressing the non linear nature of the data.

Polynomial Features (Degree=2):
Why: Simple linear regression assumes straight lines. However, medical costs often curve (e.g., costs might skyrocket as Age and BMI both increase).
The Math: We expand our equation to include squared terms and interactions: Y = b0 + b1X + b2X^2 + b3(X1 . X2)
This allows the model to fit curves rather than just straight lines.

Standard Scaler:
Why: Gradient Descent is sensitive to the scale of data. It struggles if one feature is huge (Charges: 10,000) and another is small (Age: 20).
The Math: We apply Z-score normalization: z = x-u / sigma. This forces all features to have a mean u of 0 and standard deviation sigma of 1.

SGD Regressor (Stochastic Gradient Descent):
Why: Instead of solving the math all at once (like OLS), SGD learns iteratively. It is efficient for finding the optimal weights in complex or large datasets.
The Math: The algorithm starts with random weights and updates them step by step in the opposite direction of the gradient (slope) of the loss function, slowly descending towards the minimum error.

6. Visualization of Results

Scatter Plot (Graph 4):
Plots Predicted Y axis vs. Actual X axis. The red line represents perfect prediction Y=X. Points adhering close to this line indicate high accuracy.

Residual Plot (Graph 5):
Plots the errors (Residual = Actual - Predicted). We look for random scatter around zero. If we see patterns (as a U-shape), it means our model is still missing some non linear information.
