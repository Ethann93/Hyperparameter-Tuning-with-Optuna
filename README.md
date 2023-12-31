# Hyperparameter Tuning with Optuna

In this project, we explore the process of hyperparameter tuning using the Optuna library to improve the performance of a machine learning model. Specifically, we focus on optimizing hyperparameters for a RandomForestRegressor model on a dataset related to life expectancy. 

This a model that added to my arsenal, where we test whether we achieves a better MAE & MSE scores with comparison to my other RandomForest model I carried out in the follwoing repository: https://github.com/Ethann93/Random-Forest-HealthCare/tree/7f461d5f9985edb7a9ac86294f4d5dd392b13bd3. One attractive aspect of this model is that it suggests the best hyperparameter tunings to adapt within our regressor. 

## Data Exploration

We begin by loading the dataset and conducting some exploratory data analysis to understand its structure and characteristics.

- The dataset contains information about health-related factors and life expectancy.
- It comprises both numerical and categorical features.
- The target variable is "Life Expectancy."

## Model Preparation

Before applying our manual hyperparameter tuning, we perform the following data preparation steps:

### Create Binary Numbers for Categorical Columns

We convert categorical columns into binary numbers using one-hot encoding to make them suitable for machine learning algorithms.

### Assign Features and Target

We separate the dataset into features (X) and the target variable (y). The target variable is "Life Expectancy," and the features contain all other columns.

### Split the Data

We split the dataset into training and testing sets to evaluate the model's performance.

## Model Results Prediction

To understand the model's baseline performance, we initially train a RandomForestRegressor model without hyperparameter tuning.

### Random Forest Regressor Model

We fit a RandomForestRegressor model to the training data and evaluate its performance on the test set. The scores for our performance metrics are:

- MAE: 0.3114
- MSE: 0.1553
- R^2: 0.9836

## Optuna Model

Now, we proceed to discover the best hyperparameter tuning using Optuna.

### Objective Function

We define an objective function that Optuna will optimize. This function takes various hyperparameters, including n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, and criterion, as suggestions for optimization. We use cross-validation with negative mean squared error as the scoring metric.

### Optuna Study

We create an Optuna study and optimize the objective function to find the best hyperparameters for our RandomForestRegressor model.

### Optuna Visualizations

We utilize Optuna's visualization tools to gain insights into the optimization process.

#### Optimization History Plot

The optimization history plot illustrates the progression of the optimization process, showing how the objective function value changes over trials. It helps us assess the efficiency and effectiveness of our optimization approach.

![newplot](https://github.com/Ethann93/Hyperparameter-Tuning-with-Optuna/assets/133777296/dca3361a-2901-4811-b124-02ddecbc6a97)


The Optimization History Plot can observe how the search for **hyperparameters evolved.** The plot helps in understanding whether the optimization process improved the objective function over time or if it converged quickly to a good solution. It can be useful for making decisions about the effectiveness of your hyperparameter optimization approach, such as whether you need to conduct more trials or if you've already reached a satisfactory result.

Accoroding to our graph, our model reached its best value by the 33rd trial, where after that trail the objective values stayed the same.

In summary, the plot illustrates the **progress of our hyperparameter optimization** study by showing how the objective function value changes as you conduct more trials. This can provide valuable insights into the **efficiency and effectiveness of your optimization process.**

#### Parallel Coordinate Plot

The parallel coordinate plot reveals relationships between hyperparameters and their corresponding objective function values. It assists in understanding how certain hyperparameter values are selected together and their impact on the objective function.

![newplot](https://github.com/Ethann93/Hyperparameter-Tuning-with-Optuna/assets/133777296/629692a4-8c97-4f6c-a30f-2dd236d1ceed)

The plot illustrates serval important aspects:

1) **Hyperparameter Relationships:** Each vertical axis represents a hyperparameter, and the horizontal lines connecting different axes illustrate how the values of these hyperparameters relate to each other during the optimization process. You can see how certain hyperparameter values are chosen together or in relation to one another.

2. **Objective Function Values:** The color and thickness of each line segment in the plot represent the objective function value for a specific trial or combination of hyperparameters. Darker, thicker lines often indicate better objective function values, while lighter, thinner lines correspond to poorer results.

3. **Optimal Configurations:** By visually inspecting the plot, we can identify regions of the plot where the lines converge, indicating successful configurations of hyperparameters that led to good objective function values. This helps you find optimal or promising combinations of hyperparameters.

4. **Divergence and Exploration:** The spread of lines across the plot can also indicate how widely you explored the hyperparameter search space. Tight clusters suggest that the search was focused, while scattered lines may indicate that the search space was thoroughly explored.

#### Slice Plot

Slice plots allow us to explore the relationship between specific hyperparameters and the objective function value.

![newplot](https://github.com/Ethann93/Hyperparameter-Tuning-with-Optuna/assets/133777296/34b2860f-7829-4715-9c6e-8835c3319384)

These Plots provide insights into **the relationship between specific hyperparameters and the objective function value.** It visualizes how a particular hyperparameter or a combination of hyperparameters affects the optimization process.

#### Hyperparameter Importances

This visualization highlights the importance of each hyperparameter in the optimization process.

![newplot](https://github.com/Ethann93/Hyperparameter-Tuning-with-Optuna/assets/133777296/a8aededb-df7b-42c1-aa79-db975c4b84e9)


### Best Hyperparameters

We extract the best hyperparameters from the Optuna study to use in our model.

## RandomForest Model with Best Hyperparameters

We create a new RandomForestRegressor model with the best hyperparameters and evaluate its performance on the test set. The scores for our performance metrics with the suggested hyperparameters are:

- MAE: 0.3761
- MSE: 0.2073
- R^2: 0.9781

By utilizing Optuna for hyperparameter tuning, we weren't able to improve the model's performance. We achieved lower MAE and MSE scores. However, the R2 score was as high as in previous models, suggesting that there may still be room for further model improvement or exploration of different algorithms.

In summary, this project demonstrates the process of hyperparameter tuning using Optuna, where it can potentially lead to enhancing the future model's performance.

