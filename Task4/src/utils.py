import pandas as pd
import numpy as np

def prefilter_items(data, take_n_popular=5000):
    """Предфильтрация товаров"""
    
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
     # 1. Удаление товаров, со средней ценой < 1$
    
    data = data[data['price'] > 1]
    
    # 2. Удаление товаров со соедней ценой > 30$
    
    data = data[data['price'] < 30]
    
    # 3. Отбрасываем наиболее популярные товары, потому что их и так купят    
    
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id' : 'share_unique_users'}, inplace=True)
    
    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]
    
    ##   И самые не популярные товары потому что их все равно не купят
    
    top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]
    
    # 4. Выбор топ-N самых популярных товаров (N = take_n_popular)
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top_5000 = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    
    data = data[data['item_id'].isin(top_5000)]
    
    return data