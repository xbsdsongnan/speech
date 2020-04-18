# sn
import numpy as np
from utils import *
import scipy.cluster.vq as vq

num_gaussian = 5
num_iterations = 5
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

class GMM:
    def __init__(self, D, K=5):
        assert(D>0)
        self.dim = D
        self.K = K # 混合模型中的分模型的个数
        #Kmeans Initial
        self.mu = []# 每个分模型的均值向量
        self.sigma = []# 每个分模型的协方差 
        data = read_all_data('train/feats.scp')
        (centroids, labels) = vq.kmeans2(data, K, minit="points", iter=100)
        clusters = [[] for i in range(self.K)]
        for (l,d) in zip(labels,data):
            clusters[l].append(d)

        for cluster in clusters:
            self.mu.append(np.mean(cluster, axis=0))
            self.sigma.append(np.cov(cluster, rowvar=0))
        self.pi = np.ones(self.K, dtype="double") / np.array([len(c) for c in clusters])

    
    def gaussian(self , x , mu , sigma):
    
        """Calculate gaussion probability.
    
            :param x: The observed data, dim*1.
            :param mu: The mean vector of gaussian, dim*1
            :param sigma: The covariance matrix, dim*dim
            :return: the gaussion probability, scalor
        """
        # 
        D=x.shape[0] ##
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma + 0.0001)
        mahalanobis = np.dot(np.transpose(x-mu), inv_sigma)
        mahalanobis = np.dot(mahalanobis, (x-mu))
        const = 1/((2*np.pi)**(D/2))
        return const * (det_sigma)**(-0.5) * np.exp(-0.5 * mahalanobis)
    
    def calc_log_likelihood(self , X): ##计算ＧＭＭ对数似然
        """Calculate log likelihood of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of current model 
        """
        #from scipy.stats import multivariate_normal
        
        log_llh = 0.0
        
        '''
        D = len(X)
        for n in range(D):
                temp = 0.0
                for m in range(self.K):
                    temp +=self.pi[m] * (gaussian(X[n], self.mu[m,:], self.sigma[m,:,:]))
                log_llh += np.log(temp)
        '''
        
        """
            FINISH by YOUSELF
        """
        
        for i in range(len(X)):
            cur = 0.0
            for j in range(self.K):
                cur += self.pi[j] * self.gaussian(X[i], self.mu[j], self.sigma[j])
            log_llh += np.log(cur)
        
        
        '''
        for i in range(len(X)):    
            inner_sum = 0
            for j in range(self.K):
                inner_sum += self.pi[j] * multivariate_normal.pdf(X[i], self.mu[j], self.sigma[j])
            log_llh += np.log(inner_sum)
        #return outer_sum
        '''
        
        return log_llh

    def em_estimator(self , X):
        """Update paramters of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of updated model 
        """
        log_llh = 0.0
        '''
        num_gau = len(X)  # 高斯分布个数
        num_data = X.shape[0]  # 数据个数
        gama = np.zeros((num_gau, num_data))  # gama[j][i]表示第i个样本点来自第j个高斯模型的概率
        #likelihood_record = []  # 记录每一次迭代的log-likelihood值
        for i in range(num_gau):
            for j in range(num_data):
                gama[i][j] = self.pi[i] * self.gaussian(X[j], self.mu[i], self.sigma[i]) / \
                             sum([self.pi[t] * self.gaussian(X[j], self.mu[t], self.sigma[t]) for t in range(num_gau)])
            #cur_likelihood = calc_log_likelihood(self , X)  # 计算当前log-likelihood
            #likelihood_record.append(cur_likelihood)
            # 更新mu
            for i in range(num_gau):
                mu[i] = np.dot(gama[i], data) / np.sum(gama[i])
            # 更新sigma
            for i in range(num_gau):
                cov = [np.dot((data[t] - mu[i]).reshape(-1, 1), (data[t] - mu[i]).reshape(1, -1)) for t in range(num_data)]
                cov_sum = np.zeros((2, 2))
                for j in range(num_data):
                    cov_sum += gama[i][j] * cov[j]
                sigma[i] = cov_sum / np.sum(gama[i])
            # 更新k
            for i in range(num_gau):
                k[i] = np.sum(gama[i]) / num_data
            #print('step: {}\tlikelihood:{}'.format(step + 1, cur_likelihood))
        #return k, mu, sigma, gama, likelihood_record
        '''
        """
            FINISH by YOUSELF
        """
        # E_step
        N, D = X.shape
        gama_mat = np.zeros((N, self.K))
        # 计算各个高斯模型中所有的样本出现的概率，行对应样本，列对应模型
        prob = np.zeros((N, self.K))
        for k in range(self.K):
            prob[:, k] = self.gaussian(X[k, :], self.mu[k], self.sigma[k])
        # 计算每个模型对每个样本的响应
        for k in range(self.K):
            gama_mat[:, k] = self.pi[k] * prob[:, k]

        for i in range(N):
            gama_mat[i, :] /= np.sum(gama_mat[i, :])

        # M_step
        mu = np.zeros((self.K, D))
        pi = np.zeros(self.K)
        sigma = np.zeros((self.K, D, D))

        for k in range(self.K):
            Nk = np.sum(gama_mat[:, k])
            # 更新mu
            # 对每个特征求均值
            for d in range(D):
                mu[k][d] = np.sum(np.multiply(gama_mat[:, k], X[:, d])) / Nk
            # 更新sigma
            for i in range(N):
                left = np.reshape((X[i] - mu[k]), (D, 1))
                right = np.reshape((X[i] - mu[k]), (1, D))
                sigma[k] += gama_mat[i, k] * left * right
            sigma[k] /= Nk
            pi[k] = Nk / N
        self.mu = mu
        self.sigma = sigma
        self.pi = pi
        
        log_llh = self.calc_log_likelihood(X) #重新计算
        
        return log_llh


def train(gmms, num_iterations = num_iterations):
    dict_utt2feat, dict_target2utt = read_feats_and_targets('train/feats.scp', 'train/text')
    
    for target in targets:
        feats = get_feats(target, dict_utt2feat, dict_target2utt)   #
        for i in range(num_iterations):
            log_llh = gmms[target].em_estimator(feats)
    return gmms

def test(gmms):
    correction_num = 0
    error_num = 0
    acc = 0.0
    dict_utt2feat, dict_target2utt = read_feats_and_targets('test/feats.scp', 'test/text')
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]
        for utt in utts:
            dict_utt2target[utt] = target
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])
        scores = []
        for target in targets:
            scores.append(gmms[target].calc_log_likelihood(feats))
        predict_target = targets[scores.index(max(scores))]
        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1
    acc = correction_num * 1.0 / (correction_num + error_num)
    return acc


def main():
    gmms = {}
    for target in targets:
        gmms[target] = GMM(39, K=num_gaussian) #Initial model
    gmms = train(gmms)
    acc = test(gmms)
    print('Recognition accuracy: %f' % acc)
    fid = open('acc.txt', 'w')
    fid.write(str(acc))
    fid.close()


if __name__ == '__main__':
    main()
