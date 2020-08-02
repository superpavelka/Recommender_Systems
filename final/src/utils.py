import pandas as pd
import numpy as np

def prefilter_items(data, take_n_popular=5000, item_features=None):
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features.\
                                        groupby('department')['item_id'].nunique().\
                                        sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]


    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 1]

    # Уберем слишком дорогие товарыs
    data = data[data['price'] < 30]

	# Уберем товары, которые не продавались за последние 12 месяцев
    data = data[data['week_no'] >= data['week_no'].max() - 52]
	
    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    
    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999
    
    # ...

    return data

def get_targets_sec_level(data, train, recommender, N):
    """Подготовка обучающего датасета, разбиение на X и y"""

    users_lvl_2 = pd.DataFrame(data['user_id'].unique())

    users_lvl_2.columns = ['user_id']

    train_users = train['user_id'].unique()
    users_lvl_2 = users_lvl_2[users_lvl_2['user_id'].isin(train_users)]

    # Рекомендации на основе собственных покупок
    users_lvl_2_ = users_lvl_2.copy()
    users_lvl_2['candidates'] = users_lvl_2['user_id'].apply(
        lambda x: recommender.get_own_recommendations(x, N=N)
    )

    s = users_lvl_2.apply(
        lambda x: pd.Series(x['candidates']), axis=1
    ).stack().reset_index(level=1, drop=True)

    s.name = 'item_id'

    users_lvl_2 = users_lvl_2.drop('candidates', axis=1).join(s)

    users_lvl_2['flag'] = 1

    targets_lvl_2 = data[['user_id', 'item_id']].copy()
    targets_lvl_2.head(2)

    targets_lvl_2['target'] = 1 

    targets_lvl_2 = users_lvl_2.merge(targets_lvl_2, on=['user_id', 'item_id'], how='left')

    targets_lvl_2['target'].fillna(0, inplace=True)
    targets_lvl_2.drop('flag', axis=1, inplace=True)

    return targets_lvl_2
	
def extend_new_user_features(data, user_features, users_emb_df):
    """Новые признаки для пользователей"""
	
    data['price']=data['sales_value']/data['quantity']
    new_user_features = user_features.merge(data, on='user_id', how='left')


    # Эмбеддинги
    user_features = user_features.merge(users_emb_df, how='left')


    # Standart sale time
    time = new_user_features.groupby('user_id')['trans_time'].mean().reset_index()
    time.rename(columns={'trans_time': 'mean_time'}, inplace=True)
    time = time.astype(np.float32)
    user_features = user_features.merge(time, how='left')


    # Age
    user_features['age'] = user_features['age_desc'].replace(
        {'65+': 70, '45-54': 50, '25-34': 30, '35-44': 40, '19-24':20, '55-64':60}
    )
    user_features = user_features.drop('age_desc', axis=1)


    # Income
    user_features['income'] = user_features['income_desc'].replace(
        {'35-49K': 45,
     '50-74K': 70,
     '25-34K': 30,
     '75-99K': 95,
     'Under 15K': 15,
     '100-124K': 120,
     '15-24K': 20,
     '125-149K': 145,
     '150-174K': 170,
     '250K+': 250,
     '175-199K': 195,
     '200-249K': 245}
    )
    user_features = user_features.drop('income_desc', axis=1)


    # Children 
    user_features['children'] = 0
    user_features.loc[(user_features['kid_category_desc'] == '1'), 'children'] = 1
    user_features.loc[(user_features['kid_category_desc'] == '2'), 'children'] = 2
    user_features.loc[(user_features['kid_category_desc'] == '3'), 'children'] = 3
    user_features = user_features.drop('kid_category_desc', axis=1)


    # Средний чек, средний чек в неделю
    basket = new_user_features.groupby(['user_id'])['price'].sum().reset_index()

    baskets = new_user_features.groupby('user_id')['basket_id'].count().reset_index()
    baskets.rename(columns={'basket_id': 'baskets'}, inplace=True)

    avr_bask = basket.merge(baskets)

    avr_bask['avr_bask'] = avr_bask.price / avr_bask.baskets
    avr_bask['sum_per_week'] = avr_bask.price / new_user_features.week_no.nunique()

    avr_bask = avr_bask.drop(['price', 'baskets'], axis=1)
    user_features = user_features.merge(avr_bask, how='left')

    return user_features	
	
def extend_new_item_features(data, item_features, items_emb_df):
    """Новые признаки для продуктов"""
    new_features = item_features.merge(data, on='item_id', how='left')

    # Эмбеддинги
    item_features = item_features.merge(items_emb_df, how='left')

    # manufacturer
    rare_manufacturer = item_features.manufacturer.value_counts()[item_features.manufacturer.value_counts() < 50].index
    item_features.loc[item_features.manufacturer.isin(rare_manufacturer), 'manufacturer'] = 999999999
    item_features.manufacturer = item_features.manufacturer.astype('object')

    # discount
    mean_disc = new_features.groupby('item_id')['coupon_disc'].mean().reset_index().sort_values('coupon_disc')
    item_features = item_features.merge(mean_disc, on='item_id', how='left')    

    # Среднее количество продаж товара в категории в неделю
    items_in_department = new_features.groupby('department')['item_id'].count().reset_index().sort_values(
        'item_id', ascending=False
    )
    items_in_department.rename(columns={'item_id': 'items_in_department'}, inplace=True)

    sales_count_per_dep = new_features.groupby(['department'])['quantity'].count().reset_index().sort_values(
        'quantity', ascending=False
    )
    sales_count_per_dep.rename(columns={'quantity': 'sales_count_per_dep'}, inplace=True)

    items_in_department = items_in_department.merge(sales_count_per_dep, on='department')
    items_in_department['qnt_of_sales_per_item_per_dep_per_week'] = (
            items_in_department['sales_count_per_dep'] /
            items_in_department['items_in_department'] /
            new_features['week_no'].nunique()
    )
    items_in_department = items_in_department.drop(['items_in_department'], axis=1)
    item_features = item_features.merge(items_in_department, on=['department'], how='left')

	# Количество продаж и среднее количество продаж товара
    item_qnt = new_features.groupby(['item_id'])['quantity'].count().reset_index()
    item_qnt.rename(columns={'quantity': 'quantity_of_sales'}, inplace=True)

    item_qnt['sales_count_per_week'] = item_qnt['quantity_of_sales'] / new_features['week_no'].nunique()
    item_features = item_features.merge(item_qnt, on='item_id', how='left')

    # sub_commodity_desc
    items_in_department = new_features.groupby('sub_commodity_desc')['item_id'].count().reset_index().sort_values(
        'item_id', ascending=False
    )
    items_in_department.rename(columns={'item_id': 'items_in_sub_commodity_desc'}, inplace=True)

    sales_count_per_dep = new_features.groupby(['sub_commodity_desc'])[
        'quantity'].count().reset_index().sort_values(
        'quantity', ascending=False
    )
    sales_count_per_dep.rename(columns={'quantity': 'qnt_of_sales_per_sub_commodity_desc'}, inplace=True)

    items_in_department = items_in_department.merge(sales_count_per_dep, on='sub_commodity_desc')
    items_in_department['qnt_of_sales_per_item_per_sub_commodity_desc_per_week'] = (
            items_in_department['qnt_of_sales_per_sub_commodity_desc'] /
            items_in_department['items_in_sub_commodity_desc'] /
            new_features['week_no'].nunique()
    )
    items_in_department = items_in_department.drop(['items_in_sub_commodity_desc'], axis=1)
    item_features = item_features.merge(items_in_department, on=['sub_commodity_desc'], how='left')

    return item_features	

def extend_user_item_new_features(data, train, recommender, item_features, user_features, items_emb_df, users_emb_df, N=50):

    target = get_targets_sec_level(data, train, recommender, N)
    user_features = extend_new_user_features(data, user_features, users_emb_df)
    item_features = extend_new_item_features(data, item_features, items_emb_df)
    item_features = data.merge(item_features, on='item_id', how='left')

    new_data = item_features.merge(user_features, on='user_id', how='left')

    # коэффициент количества покупок товаров в данной категории к среднему количеству покупок
    count_perch = new_data.groupby(['user_id', 'commodity_desc', 'week_no']).agg({'quantity': 'mean'}) \
        .reset_index().rename(columns={'quantity': 'count_purchases_week_dep'})

    mean_count_perch = new_data.groupby(['commodity_desc', 'week_no']).agg({'quantity': 'sum'}) \
        .reset_index().rename(columns=({'quantity': 'mean_count_purchases_week_dep'}))

    coef = count_perch.merge(mean_count_perch, on=['commodity_desc', 'week_no'], how='left')
    coef['count_purchases_week_mean'] = coef['count_purchases_week_dep'] / coef['mean_count_purchases_week_dep']
    coef = coef[['user_id', 'commodity_desc', 'count_purchases_week_mean']]

    temp = coef.groupby(['user_id', 'commodity_desc']).agg({'count_purchases_week_mean': 'mean'}) \
        .reset_index()

    new_data = new_data.merge(temp, on=['user_id', 'commodity_desc'], how='left')

    """коэффициент отношения суммы покупок товаров в данной категории к средней сумме"""
    count_perch = new_data.groupby(['user_id', 'commodity_desc', 'week_no']).agg({'price': 'sum'}) \
        .reset_index().rename(columns={'price': 'price_week'})

    mean_count_perch = new_data.groupby(['commodity_desc', 'week_no']).agg({'price': 'sum'}) \
        .reset_index().rename(columns=({'price': 'mean_price_week'}))

    coef = count_perch.merge(mean_count_perch, on=['commodity_desc', 'week_no'], how='left')
    coef['sum_purchases_week_mean'] = coef['price_week'] / coef['mean_price_week']
    coef = coef[['user_id', 'commodity_desc', 'sum_purchases_week_mean']]

    temp = coef.groupby(['user_id', 'commodity_desc']).agg({'sum_purchases_week_mean': 'mean'}) \
        .reset_index()

    new_data = new_data.merge(temp, on=['user_id', 'commodity_desc'], how='left')

    new_data = new_data.merge(target, on=['item_id', 'user_id'], how='left')
    new_data = new_data.fillna(0)

    return new_data
	
def get_important_features(model, X_train, y_train):
    """Возвращает важные фичи"""
	
    model.fit(X_train, y_train)
    feature = list(zip(X_train.columns.tolist(), model.feature_importances_))
    feature = pd.DataFrame(feature, columns=['feature', 'value'])
    features = feature.loc[feature.value > 0, 'feature'].tolist()
    return features
	
def get_popularity_recommendations(data, n=5):
    """Топ-n популярных товаров"""

    popular = data.groupby('item_id')['quantity'].count().reset_index()
    popular.sort_values('quantity', ascending=False, inplace=True)
    popular = popular[popular['item_id'] != 999999]
    recs = popular.head(n).item_id
    return recs.tolist()

def filter_by_diff_cat(list_recommendations, item_info):
    """Получение списка товаров из уникальных категорий"""
	
    final_recommendations = []

    categories_used = []

    for item in list_recommendations:
        category = item_info.loc[item_info['item_id'] == item, 'sub_commodity_desc'].values[0]

        if category not in categories_used:
            final_recommendations.append(item)
            categories_used.append(category)

    return final_recommendations


def postfilter_items(row, item_info, train_1, price, list_pop_rec, N=5):
    """Пост-фильтрация товаров
    Input
    -----
    row: строка датасета
    item_info: pd.DataFrame
        Датафрейм с информацией о товарах
     train_1: pd.DataFrame
        обучающий датафрейм
    """
    recommend = row['recomendations']
    purchased_goods = train_1.loc[train_1['user_id'] == row['user_id']]['item_id'].unique()

    if recommend == 0:
        recommend = list_pop_rec

    # Unique
    unique_recommendations = []
    [unique_recommendations.append(item) for item in recommend if item not in unique_recommendations]

    # More then 1$
    price_recommendations = []
    [price_recommendations.append(item) for item in unique_recommendations if price \
        .loc[price['item_id'] == item]['price'].values > 1]

    # 1 товар > 7 $
    expensive_items = []
    [expensive_items.append(item) for item in price_recommendations if price. \
        loc[price['item_id'] == item]['price'].values > 7]

    if len(expensive_items) ==0:
        [expensive_items.append(item) for item in list_pop_rec if price. \
            loc[price['item_id'] == item]['price'].values > 7]

    # товар который юзер не покупал
    new_items = []
    [new_items.append(item) for item in price_recommendations if item not in purchased_goods]

    # Промежуточный итог
    rec = []
    rec.append(expensive_items[0] if len(expensive_items) > 0 else list_pop_rec[0])
    rec += new_items
    rec = filter_by_diff_cat(rec, item_info=item_info)[0:3]
    rec += price_recommendations
    final_recommendations = filter_by_diff_cat(rec, item_info=item_info)

    n_rec = len(final_recommendations)
    if n_rec < N:
        final_recommendations.extend(list_pop_rec[:N - n_rec])
    else:
        final_recommendations = final_recommendations[:N]

    assert len(final_recommendations) == N, 'Количество рекомендаций != {}'.format(N)
	
    return final_recommendations

def get_final_recomendations(X_test, test_preds_proba, val_2, train_1, item_features):

    """Финальный список рекомендованных товаров"""
    X_test['predict_proba'] = test_preds_proba

    X_test.sort_values(['user_id', 'predict_proba'], ascending=False, inplace=True)
    recs = X_test.groupby('user_id')['item_id']
    recomendations = []
    for user, preds in recs:
        recomendations.append({'user_id': user, 'recomendations': preds.tolist()})

    recomendations = pd.DataFrame(recomendations)

    result_2 = val_2.groupby('user_id')['item_id'].unique().reset_index()
    result_2.columns = ['user_id', 'actual']

    result = result_2.merge(recomendations, how='left')
    result['recomendations'] = result['recomendations'].fillna(0)

    price = train_1.groupby('item_id')['price'].mean().reset_index()

    pop_rec = get_popularity_recommendations(train_1, n=500)
    list_pop_rec = []
    [list_pop_rec.append(item) for item in pop_rec if price \
        .loc[price['item_id'] == item]['price'].values > 1]

    result['recomendations'] = result.progress_apply \
        (lambda x: postfilter_items(x, item_info=item_features, train_1=train_1, price=price, list_pop_rec=list_pop_rec, N=5), axis=1)

    return result
