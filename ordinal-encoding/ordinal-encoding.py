def ordinal_encoding(values, ordering):
    index_map = {category : idx for idx, category in enumerate(ordering)}
    return [index_map[value] for value in values]