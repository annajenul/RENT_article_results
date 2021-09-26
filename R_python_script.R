library(reticulate)

# import modules
pd <- import("pandas")
np <- import("numpy")
skl_lr <- import("sklearn.linear_model")
skl_prepro <- import("sklearn.preprocessing")
metrics = import("sklearn.metrics")

classification_scores <- function(train_data, train_labels, 
                                  test_data, test_labels){
  
  if (!is.data.frame(train_data)) {
    train_data = as.matrix(train_data)
    test_data = as.matrix(test_data)
  }
  
  # convert to python objects
  py_train <- r_to_py(train_data)
  py_train_labels <- r_to_py(train_labels)
  py_test <- r_to_py(test_data)
  py_test_labels <- r_to_py(test_labels)
  
  # standardize
  sc = skl_prepro$StandardScaler()
  py_train_sc = sc$fit_transform(py_train)
  py_test_sc = sc$transform(py_test)
  
  # logistic regression model
  model = skl_lr$LogisticRegression(penalty="none",max_iter=8000, solver="saga")
  model$fit(py_train_sc, py_train_labels)
  
  # predict
  f1_1 = metrics$f1_score(py_test_labels, model$predict(py_test_sc), pos_label=1)
  f1_0 = metrics$f1_score(py_test_labels, model$predict(py_test_sc), pos_label=0)
  acc = metrics$accuracy_score(py_test_labels, model$predict(py_test_sc))
  mcc = metrics$matthews_corrcoef(py_test_labels, model$predict(py_test_sc))
  
  results = list("f1_1"=f1_1, "f1_0"=f1_0, "acc" = acc, "mcc" = mcc)
  return(results)
}


regression_scores <- function(train_data, train_labels, 
                                  test_data, test_labels){
  
  if (!is.data.frame(train_data)) {
    train_data = as.matrix(train_data)
    test_data = as.matrix(test_data)
  }
  
  # convert to python objects
  py_train <<- r_to_py(train_data)
  py_train_labels <- r_to_py(train_labels)
  py_test <- r_to_py(test_data)
  py_test_labels <- r_to_py(test_labels)
  
  # standardize
  sc = skl_prepro$StandardScaler()
  py_train_sc = sc$fit_transform(py_train)
  py_test_sc = sc$transform(py_test)
  
  
  # linear Model
  model = skl_lr$LinearRegression() 
  model$fit(py_train_sc, py_train_labels)
  
  # predict
  rmse = np$sqrt(metrics$mean_squared_error(py_test_labels, model$predict(py_test_sc)))
  r2 = metrics$r2_score(py_test_labels, model$predict(py_test_sc))
  
  results = list("rmse"=rmse, "r2"=r2)
  return(results)
}

