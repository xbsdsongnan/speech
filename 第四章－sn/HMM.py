# Author: Kaituo Xu, Fan Yu
import numpy as np

def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model #初始状态概率分布,状态转移概率分布,观察概率分布
    T = len(O) #观测序列的时刻个数T
    N = len(pi) #HMM的状态个数N
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here
    ##N,T = np.shape(A)[0], np.shape(O)[0]
    print('HMM的状态个数N=%d'%N, '观测序列的时刻总数T=%d'%T)   
    alpha = np.zeros((T,N))  #保存每个时刻每个状态的观测序列出现的前向概率
    '''
    for i in range(N): 
        alpha[0, i] = np.array(pi)[i, 0] * np.array(B)[i, 0]
    for t in range(T - 1):
        for i in range(N):
            temp_value = 0.0;
            for j in range(N):
                temp_value += alpha[t, j] * np.array(A)[j, i]
            index = 0
            if(O[t + 1, 0] == 0):
                index = 0
            else:
                index = 1
            alpha[t + 1, i] = temp_value * np.array(B)[i, index]  
    for i in range(N):
        prob += alpha[T - 1, i]
    return prob,alpha
    '''
    #alpha = np.zeros((T,N),dtype=float)
    ##print(alpha)
    '''
    alpha[0,:] = np.array(pi) * np.array(B)[:,O[0]]  #numpy可以简化循环
    for t in range(1,T):
            for n in range(0,N):
                alpha[t,n] = np.dot(alpha[t-1,:],np.array(A)[:,n]) * np.array(B)[n,O[t]] #使用内积简化代码
    return alpha
    '''
    
    for t in range(T):
        if 0 == t: #计算初值
            alpha[t] = np.multiply(np.array(pi), np.array(B)[:,O[t]])
            #alpha[t] = np.multiply(Pi[0,:], B[:,O[t]])
        else: #递推计算
            for i in range(N):
                alpha_t_i = np.multiply(alpha[t-1], np.array(A)[:,i])*np.array(B)[i, O[t]]
                alpha[t,i] = sum(alpha_t_i)
    print('alpha:\n', alpha)
    return sum(alpha[t])
    
    # End Assignment
    #return prob


def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here
    #(1)计算初值
    beta0 = np.ones((1,N)) #后向概率的初值
    #(2)递推计算
    beta = np.zeros((T,N))
    t_range = -1 * np.array(sorted(-1 * np.arange(T)))
    for t in t_range:
        if 0 == t:#终止递推，计算观测序列出现的概率
            a, b, backward = np.array(pi), np.array(B)[:, O[t]], beta[t+1]
            beta[t] = np.multiply(np.multiply(a, b),backward)
        else: #递推计算
            for i in range(N):
                a,b,backward = np.array(A)[i],np.array(B)[:, O[t]],[]
                if T-1 == t:
                    backward = beta0
                else:
                    backward = beta[t+1]
                beta_t_i = np.multiply(np.multiply(a, b),backward)
                beta[t,i] = beta_t_i.sum()
    return beta[0].sum()
    # End Assignment
    #return prob
 

def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, []
    # Begin Assignment

    # Put Your Code Here
    delta = np.zeros((T,N))#局部最优状态序列的概率数组
    psi = np.zeros((T,N))#局部最优状态序列的前导状态索引数组
    #(1)viterbi algorithm
    for t in range(T):#[0,1,...,T-1]
        if 0 == t:#计算初值
            delta[t] = np.multiply(np.array(pi).reshape((1, N)), np.array(np.array(B)[:,O[t]]).reshape((1, N)))
            continue
        for i in range(N):
            delta_t_i = np.multiply(np.multiply(delta[t-1], np.array(A)[:,i]), np.array(B)[i, O[t]])
            delta[t,i] = max(delta_t_i)
            psi[t][i] = np.argmax(delta_t_i)
    states = np.zeros((T,))
    t_range = -1 * np.array(sorted(-1*np.arange(T)))
    for t in t_range:
        if T-1 == t:
            states[t] = np.argmax(delta[t])
        else:
            states[t] = psi[t+1, int(states[t+1])]
    print('局部最优概率分布:\n', delta)
    print('局部最优前时刻状态索引:\n', psi)
    #print('最优状态序列:', states)
    #return states
    return states
    #print(sum(delta[t]))
    # End Assignment
    #return best_prob, best_path


if __name__ == "__main__":
    color2id = { "RED": 0, "WHITE": 1 }
    # model parameters
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0) #观测序列
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    #print(observ_prob_forward)
    print('HMM前向算法观测序列出现概率:',observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    #print(observ_prob_backward)
    print('HMM后向算法观测序列出现概率:', observ_prob_backward)

    #best_prob, best_path = Viterbi_algorithm(observations, HMM_model) 
    #print(best_prob, best_path)
    states = Viterbi_algorithm(observations, HMM_model) 
    print('最优状态序列:', states)
