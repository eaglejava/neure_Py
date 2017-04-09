import numpy as np

# """
# eta 学习率
# n_iter  权重向量的训练次数
# w_  神经元分叉权重向量
# errors_ 记录神经元判断出现错误的次数
# """
class Perceptron(object):
    def __init__(self ,eta=0.01,n_iter=10):
        self.eta = eta;
        self.n_iter = n_iter;
        pass
    def fit(self,X,y):
#         '''
#                              输入训练数据,培训神经元,X输入样本向量,对应样本分类
#           X:shap[n_samples,n_features]
#         '''
#         '''
#                             初始化权重向量为0 加一是因为 不调函数的阀值 
#         '''
        self.w_ = np.zeros(1+X.shape[1]);
        self.errors = [];
        
        for _ in range(self.n_iter):
            errors = 0;
            for xi,target in zip(X,y):
                update = self.eta * (target - self.predict(xi));
                self.w_[1:] += update * xi;
                #更新阈值
                self.w_[0] += update;
                
                errors += int(update != 0.0);

    def net_input(self,X):
        return np.dot(X,self.w_[1:]+self.w_[0]);
    
    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0 , 1,-1);