import numpy as np


class EvaluateBase:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    @property
    def _true(self):
        '''正样本'''
        return self.y_true == 1

    @property
    def _false(self):
        '''负样本'''
        return self.y_true == 0

    @property
    def _positive(self):
        '''判断为真'''
        return self.y_pred == 1

    @property
    def _negative(self):
        '''判定为假'''
        return self.y_pred == 0

    @property
    def P(self):
        '''正样本个数'''
        return sum(self._true)

    @property
    def N(self):
        '''负样本个数'''
        return sum(self._false)

    def __len__(self):
        '''样本总数'''
        return self.P + self.N

    @property
    def TP(self):
        '''真阳性
        将正类预测为正类
        '''
        _tp = self._true & self._positive
        return sum(_tp)

    @property
    def FN(self):
        '''伪阴性
        将正类预测为负类
        '''
        _fn = self._true & self._negative
        return sum(_fn)

    @property
    def FP(self):
        '''伪阳性
        将负类预测为正类
        '''
        _fp = self._false & self._positive
        return sum(_fp)

    @property
    def TN(self):
        '''真阴性
        将负类预测为负类
        '''
        _tn = self._false & self._negative
        return sum(_tn)

    @property
    def confusion_matrix(self):
        cm = np.array([[self.TP, self.FN],
                       [self.FP, self.TN]])
        return cm


class Evaluate(EvaluateBase):
    @property
    def accuracy(self):
        '''精度
        '''
        _acc = self.TP + self.TN
        return _acc / len(self)

    @property
    def _P(self):
        return self.TP + self.FP

    @property
    def T(self):
        return self.TP + self.FN

    @property
    def F(self):
        return self.FP + self.TN

    @property
    def _N(self):
        return self.FN + self.TN

    @property
    def precision(self):
        '''查准率
        '''
        return self.TP/self._P

    @property
    def PPV(self):
        '''陽性預測值
        Positive predictive value
        '''
        return self.precision

    @property
    def recall(self):
        '''查全率
        '''
        return self.TP/self.T

    @property
    def TPR(self):
        '''True Positive Rate
        灵敏性
        '''
        return self.recall

    @property
    def FPR(self):
        '''False Positive Rate, Fall-out
        伪阳率
        '''
        return self.FP/self.F

    @property
    def FNR(self):
        '''False Negative Rate, Miss rate
        伪阴率
        '''
        return self.FN / self.T

    @property
    def FDR(self):
        '''False discovery rate
        '''
        return self.FP/self._P

    @property
    def FOR(self):
        '''False omission rate
        '''
        self.FN/self._N

    @property
    def TNR(self):
        '''True negative rate
        特异性
        '''
        return self.TN/self.F

    @property
    def NPV(self):
        '''Negative predictive value 
        陰性預測值
        '''
        return self.TN/self._N

    @property
    def F1(self):
        _c = 1/self.precision + 1/self.recall
        return 2 / _c

    @property
    def LR_plus(self):
        '''Postitive likelihood ratio,LR+
        '''
        return self.TPR/self.FPR

    @property
    def LR_minus(self):
        '''Negative likelihood ratio, LR-
        '''
        return self.FNR/self.TNR

    @property
    def DOR(self):
        '''Diagnostic odds ratio
        '''
        return self.LR_plus/self.LR_minus

    @property
    def FA(self):
        return self.FP/self._P

    @property
    def ACR(self):
        return (self.TP+self.TN)/(self._P + self._N)
