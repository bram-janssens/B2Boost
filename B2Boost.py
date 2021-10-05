import numpy as np
import xgboost as xgb
import scipy.stats as st

beta_dist_s = st.beta.pdf(np.arange(0,1,0.004), 6, 14)

def EMPB_validation(predictions, true, clv,  f = 15, delta = 0.05):
    
    inds = np.flip(predictions.argsort())
    #empc
    result = []
    for top_ratio in np.arange(0,1,0.005):
        #inds = np.flip(predictions.argsort())
        contacted = true[true.index[inds[0:int(round(top_ratio*len(inds)))]]]
        clvs = clv[clv.index[inds[0:int(round(top_ratio*len(inds)))]]]
        emp = 0
        i = 0
        for contact in contacted:
            if contact==1: emp += sum((beta_dist_s*0.004*((1-delta)*clvs[clvs.index[i]])-f))  
            else: emp += -f-delta*clvs[clvs.index[i]]
            i += 1    
        result.append(emp)
    emp = np.max(result) 
   
    
    #emp = empc_predefined_s(clv, true, predictions)
    return(emp)

def b2boost(dtrain: xgb.DMatrix, kBoostRound,gammab2b, lr, clv_train, gamma = 0.3,  f = 15, delta = 0.05): 
  

    def objecti(predt: np.ndarray,
                    dtrain: xgb.DMatrix):
        predt = 1 / (1 + np.exp(-predt))
        true = dtrain.get_label()
        expected_profits = +f+(delta*clv_train)+true*(clv_train*(-gamma+gamma*delta+delta))
        grad = predt * (1 - predt) * expected_profits
        hess = abs((1 - 2 * predt) * grad)
        return grad, hess

    results = dict()
    model = xgb.train({'tree_method': 'hist', 'seed': 42,
               'disable_default_eval_metric': 1, 'gamma':gammab2b,'learning_rate': lr},
              dtrain=dtrain,
              num_boost_round=kBoostRound,       
              obj=objecti)

    return model

def verbrakenboost(dtrain: xgb.DMatrix, kBoostRound,gammab2b, lr, avg_clv, gamma = 0.3,  f = 15, delta = 0.05):
    
    d = delta*avg_clv
    
    weight = (d+gamma*avg_clv-gamma*d)/(-f+gamma*avg_clv-gamma*d)
    
    results = dict()
    model = xgb.train({'tree_method': 'hist', 'seed': 42,
               'disable_default_eval_metric': 1, 'gamma':gammab2b,'learning_rate': lr, 'scale_pos_weight':weight},
              dtrain=dtrain,
              num_boost_round=kBoostRound)

    return model
  
 
