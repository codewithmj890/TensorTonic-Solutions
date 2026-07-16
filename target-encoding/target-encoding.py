def target_encoding(categories, targets):
    # 1. Accumulate sums and counts for each category
    category_sums = {}
    category_counts = {}
    
    for cat, target in zip(categories, targets):
        category_sums[cat] = category_sums.get(cat, 0) + target
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
    # 2. Compute the mean for each category
    category_means = {
        cat: category_sums[cat] / category_counts[cat] 
        for cat in category_sums
    }
    
    # 3. Map the original categories to their corresponding means
    return [category_means[cat] for cat in categories]