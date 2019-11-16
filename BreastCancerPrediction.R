# Predicting Cancer Occurrence

# Business Problem Definition: Breast Cancer Prediction
# http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

## Step 1 - Collecting the Data

# Breast cancer data includes 569 cancer biopsy observations,
# each with 32 characteristics (variables). A feature is a number of
# ID, other is cancer diagnosis, and 30 are laboratory measurements
# numeric. The diagnosis is coded as "M" to indicate malignant or "B" to indicate
# indicate benign.

dados <- read.csv("dataset.csv", stringsAsFactors = FALSE)
str(dados)
View(dados)

## Step 2 - Preprocessing

# Deleting the ID column
# Regardless of machine learning method, should always be deleted
# ID variables. Otherwise, it may lead to wrong results because the ID
# can be used to only "predict" each example. Therefore, a model
# which includes an identifier may suffer from overfitting,
# and it will be very difficult to use to generalize other data.
dados$id <- NULL

# Setting the target variable labe
dados$diagnosis = sapply(dados$diagnosis, function(x){ifelse(x=="M","Malignant","Benign")})

# Many classifiers require variables to be of type Factor
table(dados$diagnosis)
dados$diagnosis <- factor(dados$diagnosis, levels = c("Benign", "Malignant"), labels = c("Benign", "Malignant"))
str(dados)

# Checking the percentage for each class
round(prop.table(table(dados$diagnosis)) * 100, digits = 1)

# Central trend measures
# We detected a scaling problem between the data, which then needs to be normalized
# The distance calculation made by kNN is dependent on the scale measurements in the input data.
summary(dados[c("radius_mean", "area_mean", "smoothness_mean")])

# Creating a normalization function
normalizar <- function(x){
  return ( (x - min(x)) / (max(x) - min(x)) )
}

# Testing normalization function - results must be identical
normalizar(c(1,2,3,4,5))
normalizar(c(10,20,30,40,50))

# Normalizing the data
dados_norm <- as.data.frame(lapply(dados[2:31], normalizar))
View(dados_norm)

## Step 3: Training the Model with k-NN

# Importing package library
install.packages("class")
library(class)

# Creating training data and test data
dados_treino <- dados_norm[1:469, ]
dados_teste <- dados_norm[470:569, ]

# Creating labels for training and test data
dados_treino_labels <- dados[1:469, 1]
dados_teste_labels <- dados[470:569, 1]
length(dados_treino_labels)
length(dados_teste_labels)

# Fitting the model
modelo_knn_v1 <- knn(train = dados_treino,
                     test = dados_teste,
                     cl = dados_treino_labels,
                     k = 21)

# The knn () function returns a factor object with predictions for each
# example in test dataset
summary(modelo_knn_v1)

## Step 4: Evaluating and Interpreting the Model

# Loading gmodels
library(gmodels)

# Creating predicted vs. current data crosstab
# We will use sample with 100 observations: length (test_data_labels)
CrossTable(x = dados_teste_labels, y = modelo_knn_v1, prop.chisq = FALSE)

# Interpreting the results
# Crosstab shows 4 possible values, representing false/true positive and negative
# We have two columns listing the original labels in the observed data
# We have two lines listing the test data labels

# We have:
# Scenario 1: Benign Cell (Observed) x Benign Cell (Predicted) - 61 cases - true positive
# Scenario 2: Malignant Cell (Observed) x Benign (Predicted) - 00 cases - false positive (model made a mistake)
# Scenario 3: Benign Cell (Observed) x Malignant (Predicted) - 02 cases - false negative (model made a mistake)
# Scenario 4: Malignant Cell (Observed) x Malignant (Predicted) - 37 cases - true negative

# Reading the Confusion Matrix (Perspective of whether or not to have the disease)

# True Negative = Our model predicted that the person did NOT have the disease and the data showed that the person did not actually have the disease.
# False Positive = Our model predicted that the person had the disease and data showed that NO, the person had the disease.
# False Negative = Our model predicted that the person did NOT have the disease and data showed that YES, the person had the disease.
# True Positive = our model predicted that the person had the disease and data showed that YES the person had the disease

# False Positive - Type I Error
# False Negative - Type II Error

# Model performance: 98% (hit 98 out of 100)


## Step 5: Optimizing Model Performance

# Using scale() function to standardize z-score
dados_z <- as.data.frame(scale(dados[-1]))

# Confirming successful transformation
summary(dados_z$area_mean)

# Creating new training and test datasets
dados_treino <- dados_z[1:469, ]
dados_teste <- dados_z[470:569, ]
dados_treino_labels <- dados[ 1:469, 1]
dados_teste_labels <- dados[ 470:569, 1]


# Reclassifying
modelo_knn_v2 <- knn(train = dados_treino,
                     test = dados_teste,
                     cl = dados_treino_labels,
                     k = 21)
summary(modelo_knn_v2)

# Creating predicted vs. current data crosstab
CrossTable(x = dados_teste_labels, y = modelo_knn_v2, prop.chisq = FALSE)

# Try different values for k
for (i in 1:20){
  print(i)
  modelo_knn_v2 <- knn(train = dados_treino,
                       test = dados_teste,
                       cl = dados_treino_labels,
                       k = i)
  CrossTable(x = dados_teste_labels, y = modelo_knn_v2, prop.chisq = FALSE)
}

## Step 6: Building a Model with Support Vector Machine (SVM) Algorithm

# Prepare the dataset
dados <- read.csv("dataset.csv", stringsAsFactors = FALSE)
dados$id <- NULL
dados[,"index"] <- ifelse(runif(nrow(dados)) < 0.8, 1, 0)
View(dados)

# Training and test data
trainset <- dados[dados$index == 1,]
testset <- dados[dados$index == 0,]

# Getting the the index
trainColNum <- grep("index", names(trainset))

# Remove dataset index
trainset <- trainset[,-trainColNum]
testset <- testset[,-trainColNum]

# Get target variable column index from dataset
typeColNum <- grep("diag", names(dados))

# Training the model
# We set the kernel to radial as this dataset does not have a
# linear plane that can be drawn
library(e1071)
modelo_svm_v1 <- svm(diagnosis ~ .,
                     data = trainset,
                     type = "C-classification",
                     kernel = "radial")
# Forecasts

# Training data predictions
pred_train <- predict(modelo_svm_v1, trainset)

# Percentage of correct forecasts with training dataset
mean(pred_train == trainset$diagnosis)

# Test data predictions
pred_test <- predict(modelo_svm_v1, testset)

# Percentage of correct forecasts with test dataset
mean(pred_test == testset$diagnosis)

# Confusion Matrix
table(pred_test, testset$diagnosis)


## Step 7: Building a model with Random Forest Algorithm

# Training the model
library(rpart)
modelo_rf1_v1 <- rpart(diagnosis ~ ., data = trainset, 
                       control = rpart.control(cp = 0.005))

# Test data prediction
tree_pred <- predict(modelo_rf1_v1, testset, type = 'class')

# Percentage of correct forecasts with test dataset
mean(tree_pred == testset$diagnosis)

# Confusion Matrix
table(tree_pred, testset$diagnosis)