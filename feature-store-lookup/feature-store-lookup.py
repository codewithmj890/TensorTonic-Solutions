def feature_store_lookup(feature_store, requests, defaults):
    results = []
    for req in requests:
        offline = feature_store.get(req["user_id"], defaults)
        combined = {**offline, **req["online_features"]}
        results.append(combined)
    return results