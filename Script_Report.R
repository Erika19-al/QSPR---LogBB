

# Load required libraries
library(readxl)
library(caret)
library(pls)
library(ggplot2)
library(dplyr)
library(GGally)
# Load the data
train_data <- read_excel("Application_training.xls")
train_data <- na.omit(train_data)
test_data <- read_excel("Application_test.xls")

# Explore the data
summary(train_data)

ggplot(train_data, aes(x = Y)) +
  geom_histogram(aes(y = ..density..), bins = 20, color = "black", fill = "lightblue", alpha = 0.6) +
  geom_density(color = "darkblue", size = 1) +
  labs(title = "Distribution of Y", x = "Y", y = "Density") +
  theme_minimal()


summary(test_data)

 # For correlation matrix visualization

# Scatter plot for Y vs cLogP with regression line
ggplot(train_data, aes(x = cLogP, y = Y)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(title = "Scatter Plot: Y vs cLogP", x = "cLogP", y = "Y") +
  theme_minimal()

# Scatter plot for Y vs PSA with regression line
ggplot(train_data, aes(x = PSA, y = Y)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(title = "Scatter Plot: Y vs PSA", x = "PSA", y = "Y") +
  theme_minimal()

# Scatter plot for Y vs mLogP with regression line
ggplot(train_data, aes(x = mLogP, y = Y)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(title = "Scatter Plot: Y vs mLogP", x = "mLogP", y = "Y") +
  theme_minimal()

# Visualize correlations among all numerical variables in train_data
GGally::ggpairs(train_data, columns = c("Y", "cLogP", "PSA", "mLogP"),
                title = "Correlation Matrix and Scatter Plots",
                upper = list(continuous = wrap("cor", size = 3)))


# Calculate the correlation coefficient
correlation <- cor(train_data$cLogP, train_data$mLogP, method = "pearson")  # Use "spearman" if needed

# Plot
ggplot(train_data, aes(x = cLogP, y = mLogP)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = paste("Scatter Plot with Correlation: ", round(correlation, 2)),
       x = "cLogP",
       y = "mLogP") +
  theme_minimal()

# Build QSPR models
# Model 1: PSA only
model1 <- lm(Y ~ PSA, data = train_data)
summary(model1)
cat("Model 1 (PSA): R-squared =", summary(model1)$r.squared, "\n")
# Model 2: cLogP only 
model2 <- lm(Y ~ cLogP, data = train_data)
summary(model2)
cat("Model 2 (cLogP): R-squared =", summary(model2)$r.squared, "\n")

# Model 3: cLogP and PSA
model3 <- lm(Y ~ cLogP + PSA, data = train_data)
summary(model3)

# Model 4: Normalized cLogP, and PSA
min_max_scale_custom <- function(x, min_range = -1, max_range = 1) {
  ((x - min(x)) / (max(x) - min(x))) * (max_range - min_range) + min_range
}

# Apply custom min-max scaling to `cLogP` and `PSA`
train_data$ncLogP <- min_max_scale_custom(train_data$cLogP, -1, 1)
train_data$nPSA <- min_max_scale_custom(train_data$PSA, -1, 1)


model4 <- lm(Y ~ nPSA  + ncLogP, data = train_data)
summary(model4)

# Select the optimal model
# Use permutation test to assess model significance
library(permute)
set.seed(123)
permute_model <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
model <- train(Y ~ nPSA + ncLogP, data = train_data, method = "lm", trControl = permute_model)

print(model)

# Apply custom min-max scaling to `cLogP` and `PSA`
test_data$ncLogP <- min_max_scale_custom(test_data$cLogP, -1, 1)
test_data$nPSA <- min_max_scale_custom(test_data$PSA, -1, 1)


# Predict on the test set using the optimal model
test_data$predicted_logBB <- predict(model, newdata = test_data)

# Calculate R^2 for predictions
observed <- test_data$'Expt log BB' 
predicted <- test_data$predicted_logBB
R2 <- cor(observed, predicted)^2
cat("R-squared on training set:", R2, "\n")

# Plot Yexp vs Ycalc
ggplot(test_data, aes(x = observed, y = predicted)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Yexp vs Ycalc for log BB", x = "Observed log BB", y = "Predicted log BB") +
  theme_minimal()
# Install and load the RWeka package if not already installed
if (!require("RWeka")) install.packages("RWeka")
library(RWeka)




model_tree <- M5P(Y ~ PSA + cLogP + mLogP, data = train_data, control = Weka_control(N = TRUE, M = 3))


summary(model_tree)

test_data$predicted_logBB_tree <- predict(model_tree, newdata = test_data)


# Calculate Mean Squared Error (MSE)
mse <- mean((test_data$'Expt log BB' - test_data$predicted_logBB_tree )^2)

# Calculate R-squared
r_squared <- cor(test_data$'Expt log BB', test_data$predicted_logBB_tree)^2

# Print the evaluation metrics
cat("Mean Squared Error:", mse, "\n")
cat("R-squared:", r_squared, "\n")



