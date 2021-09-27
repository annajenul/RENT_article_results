library(stabs)
library(glmnet)
library(mltools)
library(caret)


stabsel_gridsearch <- function(train_data, train_labels, B=100){ # we set B=100 equal to the number of ensemble models in RENT
  set.seed(0)
  # define grid search values
  PFER_values = seq(0.05,0.95,0.05)
  cutoff_values = seq(0.6,0.9,0.05)
  folds <- createFolds(train_labels, k=5, list=TRUE)
  
  # define matrix to store average performance values over 5 folds
  mcc_average_scores = matrix(nrow=length(PFER_values), ncol = length(cutoff_values))
  rownames(mcc_average_scores)=PFER_values
  colnames(mcc_average_scores)=cutoff_values
   
  # loop over hyperparameters
  for (i in 1:length(PFER_values)) {
    for (j in 1:length(cutoff_values)) {
      cat("pfer:",PFER_values[i], "cutoff:", cutoff_values[j], "\n")
      
      # store performance for each fold
      average_performance = c()
      for (fold in folds) {
        # apply stabsel on standardized data
        stab.glmnet = stabsel(x = apply(train_data[-fold,], 2, function(x){if(length(unique(x))>1){return(scale(x))} else{return(x)}}), 
                              y = train_labels[-fold],
                              fitfun = glmnet.lasso,
                              args.fitfun = list(family = "binomial"), # binomial model for classification
                              cutoff = cutoff_values[j],
                              PFER = PFER_values[i],
                              B = B,
                              sampling.type="MB")
        
        selected_features = stab.glmnet$selected
        cat("LENGHT SELECTED FEATURES:", length(selected_features), "\n")
        
        if (length(selected_features) == 0) {
          average_performance = c(average_performance, NA)
        }
        else{
          # standardize data
          mean_train = apply(train_data[-fold,], 2, mean)
          std_train = apply(train_data[-fold,], 2, sd)
          train_data_sc = as.matrix(sweep(
            sweep(train_data[-fold,],2, mean_train),
            2, std_train, "/"))
          
          
          test_data_sc = as.matrix(sweep(
            sweep(train_data[fold,],2, mean_train),
            2, std_train, "/"))
          
          
          train_data_sc = train_data_sc[,names(selected_features)]
          test_data_sc = test_data_sc[,names(selected_features)]
          
          if (is.vector(train_data_sc)) {
            train_data_sc = data.frame(train_data_sc)
            colnames(train_data_sc) = names(selected_features)
          }
          
          glm_data = data.frame(cbind("y"=train_labels[-fold], train_data_sc))
          glm.fit <- glm(y~., data=glm_data, family = binomial)
          pred_data = data.frame(test_data_sc)
          colnames(pred_data) = colnames(train_data_sc)
          glm.pred = 1*(predict(glm.fit, newdata = pred_data,
                                type="response") >= 0.5)
          
          average_performance = c(average_performance, 
                                  mltools::mcc(preds=glm.pred, actual=train_labels[fold]))
        }
        if (all(is.na(average_performance))) {
          mcc_average_scores[i,j] = NA
        }
        mcc_average_scores[i,j] = mean(average_performance, na.rm=TRUE)
        
      }
    }
    
  }
  
  print(mcc_average_scores)
  
  # find hyperparameters leading to the best MCC values;
  params = which(mcc_average_scores == max(mcc_average_scores, na.rm=TRUE), arr.ind = TRUE)
  pfer = PFER_values[min(params[,1])]
  cutoff = cutoff_values[max(params[params[,1]==min(params[,1])][-1])]
    
  stab.glmnet = stabsel(x = apply(train_data, 2, function(x){if(length(unique(x))>1){return(scale(x))} else{return(x)}}), 
                        y = train_labels,
                        fitfun = glmnet.lasso,
                        args.fitfun = list(family = "binomial"),
                        cutoff = cutoff,
                        PFER = pfer,
                        B = B)
  
  return(stab.glmnet$selected)
}
