def linear_interpolation(values):
    result = list(values)
    n = len(result)
    i = 0
    
    while i < n:
        if result[i] is None:
            left = i - 1
            right = i
            while result[right] is None:
                right += 1
            
            v_left = result[left]
            v_right = result[right]
            span = right - left
            
            for j in range(left + 1, right):
                result[j] = v_left + (j - left) / span * (v_right - v_left)
            
            i = right
        else:
            i += 1
    
    return result