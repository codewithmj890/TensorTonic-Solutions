def matrix_factorization_sgd_step(U, V, r, lr, reg):
   
    dot_product = sum(u * v for u, v in zip(U, V))
    
    e = r - dot_product
    
    U_new = []
    V_new = []
    
    for u, v in zip(U, V):
        u_updated = u + lr * (e * v - reg * u)
        v_updated = v + lr * (e * u - reg * v)
        
        U_new.append(u_updated)
        V_new.append(v_updated)
        
    return U_new, V_new