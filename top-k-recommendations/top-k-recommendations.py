def top_k_recommendations(scores, rated_indices, k):
    unrated_items = [
        (scores[i], i) 
        for i in range(len(scores)) 
        if i not in rated_indices
    ]
    
    unrated_items.sort(key=lambda x: x[0], reverse=True)

    return [item[1] for item in unrated_items[:k]]