def item_cf_predict(user_ratings, item_similarities, target):
    weighted_sum = 0.0
    weight_total = 0.0
    
    for i in range(len(user_ratings)):
        if i == target:
            continue
        s = item_similarities[i]
        r = user_ratings[i]
        if s > 0 and r != 0:
            weighted_sum += s * r
            weight_total += s
    
    if weight_total == 0:
        return 0.0
    
    return weighted_sum / weight_total