from collections import defaultdict
import numpy as np

def prediction_error_lfm(alpha, beta_u, beta_i, gamma_u, gamma_i, u, i, user_ratings):
    predicted = alpha
    u_known = False
    i_known = False
    if u in beta_u:
        predicted += beta_u[u]
        u_known = True
    if i in beta_i:
        predicted += beta_i[i]
        i_known = True
    if u_known and i_known:
        predicted += np.dot(gamma_u[u], gamma_i[i])
    return (user_ratings[u][i] - predicted) ** 2

def regularization_lfm(beta_u, beta_i, gamma_u, gamma_i, l_reg):
    cost = 0.0
    for u in beta_u:
        cost += beta_u[u] ** 2
        cost += np.dot(gamma_u[u], gamma_u[u])
    for i in beta_i:
        cost += beta_i[i] ** 2
        cost += np.dot(gamma_i[i], gamma_i[i])
    return l_reg * cost

def objective_lfm(alpha, beta_u, beta_i, gamma_u, gamma_i, user_ratings, l_reg):
    error = 0.0
    for u in user_ratings:
        for i in user_ratings[u]:
            error += prediction_error_lfm(alpha, beta_u, beta_i, gamma_u, gamma_i, u, i, user_ratings)
    if l_reg == 0.0:
        reg_cost = 0.0
    else:
        reg_cost = regularization_lfm(beta_u, beta_i, gamma_u, gamma_i, l_reg)
    return error + reg_cost

def train_lfm(user_ratings, item_ratings, l_reg = 1.0, K = 1, max_iter = 25, delta = 1e-2):
    alpha = 0.0
    beta_u = defaultdict(float)
    beta_i = defaultdict(float)
    gamma_u = {}
    gamma_i = {}
    for u in user_ratings:
        gamma_u[u] = np.random.rand(K)
    for i in item_ratings:
        gamma_i[i] = np.random.rand(K)
    
    prev_cost = objective_lfm(alpha, beta_u, beta_i, gamma_u, gamma_i, user_ratings, l_reg)
    print "Initial cost: ", prev_cost
    
    N = 0
    I = defaultdict(int)
    U = defaultdict(int)
    for u in user_ratings:
        for i in user_ratings[u]:
            N += 1
            I[u] += 1
            U[i] += 1
    
    for ii in xrange(max_iter):
        
        # Learn alpha:
        alpha = 0.0
        for u in user_ratings:
            for i in user_ratings[u]:
                alpha += (user_ratings[u][i] - beta_u[u] - beta_i[i] - np.dot(gamma_u[u], gamma_i[i]))
        alpha /= N
        
        # Learn beta_u:
        beta_u = defaultdict(float)
        for u in user_ratings:
            for i in user_ratings[u]:
                beta_u[u] += (user_ratings[u][i] - alpha - beta_i[i] - np.dot(gamma_u[u], gamma_i[i]))
            beta_u[u] /= (l_reg + I[u])
        
        # Learn beta_i:
        beta_i = defaultdict(float)
        for i in item_ratings:
            for u in item_ratings[i]:
                beta_i[i] += (item_ratings[i][u] - alpha - beta_u[u] - np.dot(gamma_u[u], gamma_i[i]))
            beta_i[i] /= (l_reg + U[i])
            
        # Learn gamma_u:
        for u in user_ratings:
            base = (alpha + beta_u[u])
            for k in xrange(K):
                gamma_u[u][k] = 0.0
                numerator = 0.0
                denominator_sum = 0
                for i in user_ratings[u]:
                    numerator += gamma_i[i][k] * (user_ratings[u][i] - base - beta_i[i] - np.dot(gamma_u[u], gamma_i[i]))
                    denominator_sum += gamma_i[i][k] ** 2
                gamma_u[u][k] = numerator / (l_reg + denominator_sum)

        # Learn gamma_i:
        for i in item_ratings:
            base = (alpha + beta_i[i])
            for k in xrange(K):
                gamma_i[i][k] = 0.0
                numerator = 0.0
                denominator_sum = 0
                for u in item_ratings[i]:
                    numerator += gamma_u[u][k] * (item_ratings[i][u] - base - beta_u[u] - np.dot(gamma_u[u], gamma_i[i]))
                    denominator_sum += gamma_u[u][k] ** 2
                gamma_i[i][k] = numerator / (l_reg + denominator_sum)
    
        cost = objective_lfm(alpha, beta_u, beta_i, gamma_u, gamma_i, user_ratings, l_reg)
        error = objective_lfm(alpha, beta_u, beta_i, gamma_u, gamma_i, user_ratings, 0.0)
        print "Iteration #", ii, ", Cost: ", cost, ", Training Error:", error
        if abs(cost - prev_cost) < delta:
            break
        prev_cost = cost
    
    return alpha, beta_u, beta_i, gamma_u, gamma_i, cost
