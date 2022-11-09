import numpy as np
from scipy.spatial import distance

omega = 0
t = 10
unusable_value = -1

def compute_nearest_distance(p_alpha, P):
    
    nearest_dist = np.Infinity
    x_alpha, y_alpha = p_alpha
    
    if x_alpha > omega or y_alpha > omega:
        for x_beta, y_beta in P:
            if x_beta > omega or y_beta > omega:
                dist = distance.euclidean((x_alpha, y_alpha), (x_beta, y_beta))
                if dist <= nearest_dist:
                    nearest_dist = dist
                    
    if nearest_dist == np.Infinity:
        nearest_dist = unusable_value
        
    return nearest_dist

def compute_motion_distance(p_alpha, p_beta):
    
    x_alpha, y_alpha = p_alpha
    x_beta, y_beta = p_beta
    
    if ((x_alpha > omega) or (y_alpha > omega)) and ((x_beta > omega) or (y_beta > omega)):
        dist = distance.euclidean(p_alpha, p_beta)
    else:
        dist = unusable_value
    
    return dist

def compute_motion_velocity(p_alpha, p_beta):
    
    x_alpha, y_alpha = p_alpha
    x_beta, y_beta = p_beta
    
    s = compute_motion_distance(p_alpha, p_beta)
    
    if s == unusable_value:
        return unusable_value
    else:
        return s / t

def compute_motion_acceleration(p_alpha, p_beta, p_gamma):

    v_final = compute_motion_velocity(p_alpha, p_beta)
    v_initial = compute_motion_velocity(p_beta, p_gamma)

    if v_final == unusable_value or v_initial == unusable_value:
        return unusable_value
    else:
        a = (v_initial - v_final) / t
        return a
    
def is_appear(x, y):
    if x > omega or y > omega:
        return 1
    return 0