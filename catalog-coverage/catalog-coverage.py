def catalog_coverage(recommendations, n_items):
    if n_items == 0:
        return 0.0

    unique_item = set()
    for rec_items in recommendations:
        unique_item.update(rec_items)
    return len(unique_item) / n_items