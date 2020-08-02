import numpy as np

def hit_rate(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    hit_rate = (flags.sum() > 0) * 1
    
    return hit_rate
	
def hit_rate_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list[:k])
    
    hit_rate = (flags.sum() > 0) * 1
    
    return hit_rate
	
def precision(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)  # [False, False, True, True]
    
    precision = flags.sum() / len(recommended_list)
    
    return precision
	
def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision
	
def money_precision_at_k(recommended_list, bought_list, price, k=5):

    prices_recommended = []

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])

    [prices_recommended.append(price.loc[price['item_id'] == item]['price'].values) for item in
     recommended_list]

    prices_recommended = np.array(prices_recommended[:k])
    flags = np.isin(recommended_list, bought_list)
    prices = flags @ prices_recommended
    precision = prices.sum() / prices_recommended.sum()

    return precision
	
def recall(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)  # [False, False, True, True]
    
    recall = flags.sum() / len(bought_list)
    
    return recall
	
def recall_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list[:k]) 
    
    recall = flags.sum() / len(bought_list)
    
    return recall
	
def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list = recommended_list[:k]
    flags = np.isin(bought_list, recommended_list) 
    
    recall = np.dot(flags,prices_bought) / np.dot(np.ones(len(flags)), prices_bought)
    
    return recall
	
def reciprocal_rank(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list) 
    
    indexes = np.where(flags==1)
    result = 1/indexes[0][0]
    
    return result
	
def mean_reciprocal_rank(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    
    flags = np.isin(bought_list, recommended_list) 
    indexes = np.where(flags==1)
    result = sum(1/indexes[0])/len(recommended_list)
    
    return result