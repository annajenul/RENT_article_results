import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, matthews_corrcoef, accuracy_score

def BIC_hyperparameter_search(model, parameters, test, test_target):

    sc = StandardScaler()
    logloss = np.zeros(shape=(len(parameters['t1']), len(parameters['t2']),
                             len(parameters['t3'])))
    # accuracy train
    acc_train = np.zeros(shape=(len(parameters['t1']), len(parameters['t2']),
                             len(parameters['t3'])))
    # accuracy test
    acc_test = np.zeros(shape=(len(parameters['t1']), len(parameters['t2']),
                             len(parameters['t3'])))
    # matthews correlation coefficient train
    mcc_train = np.zeros(shape=(len(parameters['t1']), len(parameters['t2']),
                             len(parameters['t3'])))
    # matthews correlation coefficient test
    mcc_test = np.zeros(shape=(len(parameters['t1']), len(parameters['t2']),
                             len(parameters['t3'])))
    # Akaike information criterion
    AIC = np.zeros(shape=(len(parameters['t1']), len(parameters['t2']),
                             len(parameters['t3'])))
    # Bayesian information criterion
    BIC = np.zeros(shape=(len(parameters['t1']), len(parameters['t2']),
                             len(parameters['t3'])))
    # number of model parameters
    num_pars = np.zeros(shape=(len(parameters['t1']), len(parameters['t2']),
                             len(parameters['t3'])))
    
    # grid search t1, t2, t3
    for i, t1 in enumerate(parameters['t1']):
        for j, t2 in enumerate(parameters['t2']):
            for k, t3 in enumerate(parameters['t3']):
                sel_feat = model.select_features(t1, t2, t3)

                train_data = sc.fit_transform(model._data.iloc[:,sel_feat])

                test_data = sc.transform(test.copy().iloc[:,sel_feat])
                lr = LogisticRegression().fit(train_data, model._target)
                num_params = len(np.where(lr.coef_ != 0)[1]) + 1
                num_pars[i,j,k]=num_params
                pred_proba = lr.predict_proba(train_data)
                pred = lr.predict(train_data)
                pred_test = lr.predict(test_data)
                
                log_lik = log_loss(y_true=model._target, y_pred=pred_proba, normalize=False)
                logloss[i,j,k]=log_lik
                acc_train[i,j,k] = accuracy_score(y_true=model._target, y_pred=pred)
                mcc_train[i,j,k] = matthews_corrcoef(y_true=model._target, y_pred=pred)
                acc_test[i,j,k] = accuracy_score(y_true=test_target, y_pred=pred_test)
                mcc_test[i,j,k] = matthews_corrcoef(y_true=test_target, y_pred=pred_test)
                A = 2 * log_lik + 2 * num_params
                B = 2 * log_lik + np.log(len(pred)) * num_params
                
                AIC[i,j,k] = A
                BIC[i,j,k] = B
                
    
    results = {"AIC":AIC, "BIC":BIC, "logloss":logloss, "acc_train":acc_train,
               "acc_test":acc_test, "mcc_train":mcc_train, "mcc_test": mcc_test,
               "num params":num_pars}
    return results
                