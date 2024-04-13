# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:02:25 2024

@author: YUNYING
"""

from abc import ABC, abstractmethod
from itertools import product


DEBUG=False

class Strategy(ABC):
    """抽象策略类"""

    @abstractmethod
    def apply(self, sequence):
        """应用策略到数列"""
        pass


class SimplifyStrategy(Strategy):
    """简化策略"""
    pass


from math import gcd,sqrt

class GCDSimplifyStrategy(SimplifyStrategy):
    """最大公因数简化策略"""

    def apply(self, sequence):
        # 计算所有元素的最大公因数
        divisor = sequence[0]
        for num in sequence[1:]:
            divisor = gcd(divisor, num)
        # 用最大公因数除每个元素
        return [num // divisor for num in sequence]
    
    
class MutiplySelfStrategy(SimplifyStrategy):
    """平方策略"""
    name="平方策略"
    def apply(self, sequence):
        return [x*x for x in sequence]

class SqrtStrategy(SimplifyStrategy):
    
    name="开根号策略"
    def apply(self, sequence):
        
        res=[int(sqrt(x)) for x in sequence]
        if [i**2 for i in res]==sequence:
            return res
        else:
            raise Exception("开根号是小数")


class DecXMutiplySelfStrategy(SimplifyStrategy):
    """减X方策略"""
    
    def __init__(self,p):
        self.p=p
        self.name="减X方策略 %d"%p
        
    def apply(self, sequence):
        res=[]
        for x,i in zip(sequence,range(1,len(sequence)+1)):
            _=x-i**self.p
            if _<0:
                raise Exception("不能减X方")
            res.append(_)
        return res


    
class InsertDotStrategy(SimplifyStrategy):
    def __init__(self,p=1):
        self.p=p
        self.name="插逗号策略 %d"%p
        
    def apply(self, sequence):
        res=[]
        for num in sequence:
            tmp=str(num)
            first=float(tmp[:self.p])
            last=float(tmp[self.p:])
            res+=[first,last]
        return res


    
class NoneStrategy(SimplifyStrategy):
    """跳过策略"""
    name=""
    def apply(self, sequence):
        return sequence


class OddTermStrategy(SimplifyStrategy):
    """取奇数项策略"""
    name="取奇数项策略"

    def apply(self, sequence):
        return sequence[::2]


class EvenTermStrategy(SimplifyStrategy):
    """取偶数项策略"""
    name="取偶数项策略"

    def apply(self, sequence):
        return sequence[1::2]
    
    
class OddEvenDecStrategy(SimplifyStrategy):
    """偶奇差数项策略"""
    # name="偶奇差数项策略"
    def __init__(self,p):
        self.p=p
        self.name="偶奇差数项策略 %d"%p
        #如，第二位减几位，取值1或-1,-1以下先不管了

    def apply(self, sequence):
        if len(sequence) <6:
            raise Exception("项数太少没必要")
        keep_odd=None
        if self.p==1:
            keep_odd=True
        elif self.p==-1:
            keep_odd=False
            
        if keep_odd:
            if len(sequence) % 2 != 0:
                # sequence = sequence[:-1]
                raise Exception("不太可能")
            
            even_terms = sequence[::2]  # 奇数项
            odd_terms = sequence[1::2]  # 偶数项
            return [odd-even  for even, odd in zip(even_terms, odd_terms)]
        else:
            pad=[]
            if len(sequence) % 2 != 0:
                pad=[sequence[0]]
                sequence = sequence[1:]
            else:
                raise Exception("不支持")
            even_terms = sequence[1::2]  # 奇数项
            odd_terms = sequence[0::2]  # 偶数项
            return pad+[odd-even  for even, odd in zip(even_terms, odd_terms)]
                


# class EvenOddDecStrategy(SimplifyStrategy):
#     """奇偶差数项策略"""

#     def apply(self, sequence):
#         if len(sequence) % 2 == 0:
#             sequence = sequence[:-1]
#         odd_terms = sequence[::2]  # 奇数项
#         even_terms = sequence[1::2]  # 偶数项
#         return [odd - even for odd, even in zip(odd_terms, even_terms)]

class PrimeFactorizationSimplifyStrategy(SimplifyStrategy):
    """质因数分解简化策略"""
    name='质因数分解策略'
    def apply(self, sequence):
        if len(sequence)<3:
            raise Exception("长度太短，不应该分解")
        simplified_sequence = []
        for num in sequence:
            factors = self._prime_factorization(num)
            if len(factors)==0:
                raise Exception("无法分解")
            # 这里需要根据具体情况决定如何处理质因数
            # 例如，可以选择保留最大的两个质因数，或者将所有质因数相乘...
            factors.sort(reverse=True)
            simplified_sequence+=factors[0]
            
            # simplified_sequence.append(factors)
        return simplified_sequence

    def _prime_factorization(self, num):
        """质因数分解"""
        factors = []
        i = 2
        while i * i <= num:
            if num % i == 0:
                factors.append([i,num//i])
                # num //= i
            i += 1
        # if num > 1:
        #     factors.append(num)
        return factors
    
# class KeepLargestTwoFactorsStrategy(PrimeFactorizationSimplifyStrategy):
#     """保留最大两个质因数"""

#     def apply(self, sequence):
#         simplified_sequence = []
#         for num in sequence:
#             factors = self._prime_factorization(num)
#             factors.sort(reverse=True)  # 降序排列
#             simplified_sequence.append(factors[:2])  # 取最大的两个质因数
#         return simplified_sequence



class DifferenceStrategy(SimplifyStrategy):
    """相邻项作差策略"""

    name="相邻项作差策略"
    def apply(self, sequence):
        return [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]

class NeiborDivStrategy(SimplifyStrategy):

    name="相邻项相除策略"
    def apply(self, sequence):
        return [sequence[i+1] / sequence[i] for i in range(len(sequence)-1)]


class NeiborAddStrategy(SimplifyStrategy):
    """相相邻项相加策略"""

    name="相邻项相加策略"
    def apply(self, sequence):
        return [sequence[i+1] + sequence[i] for i in range(len(sequence)-1)]


class ValidateStrategy(Strategy):
    """验证策略"""
    pass


class CompositeValidateStrategy(Strategy):
    """复合验证策略"""

    def __init__(self, strategies,minlen=None):
        self.strategies = strategies
        self.minlen=minlen

    def apply(self, sequenceA,sequenceB):
        # s=sequence.copy()
        try:
            # for strategys in self.strategies:
            #     # sequence = strategy.apply(sequence)
            #     if all(strategy.apply(sequence) for strategy in strategys):
            #         return True
            
            
            
            def call_my_function(fun,arg1, minlen=None):
                if minlen is not None:
                    return fun(arg1, minlen=minlen)
                else:
                    return fun(arg1)
                    
            Succ_A=False
            for strategyA in self.strategies:
                # sequence = strategy.apply(sequence)
                # if minlen:
                if call_my_function(strategyA.apply,sequenceA,self.minlen):
                    # if DEBUG:
                    #     print('CompositeValidateStrategy A Succ',strategyA.name)
                    Succ_A=True
                    
                    break
                    
            if Succ_A:
                for strategyB in self.strategies:
                    if call_my_function(strategyB.apply,sequenceB,self.minlen):
                        return True,'，'.join([strategyA.name,strategyB.name])
            return False,''
            # return all(strategy.apply(sequence))
        except:
            return False,''
        


class SplitValidateStrategy(ValidateStrategy):
    """拆分验证"""

    name='验证_奇偶列拆分验证'
    def apply(self, sequence,minlen=6):
        if len(sequence) <minlen:
            return False
        res,name=CompositeValidateStrategy(validate_strategies,3).apply(sequence[0::2],sequence[1::2])
        self.name=name='验证_奇偶列拆分验证[%s]'%name
        return res
    




class ArithmeticSequenceStrategy(ValidateStrategy):
    """验证是否为等差数列"""

    name='验证_等差数列'
    def apply(self, sequence,minlen=3):
        if len(sequence) <minlen:
            return False
        d = sequence[1] - sequence[0]
        if d==0:
            return False
        return all(sequence[i+1] - sequence[i] == d for i in range(1, len(sequence)-1))

# class WeakArithmeticSequenceStrategy(ValidateStrategy):
#     """验证是否为等差数列"""

#     name='验证_弱等差数列  不能确定规律'
#     def apply(self, sequence,minlen=4):
#         if len(sequence) <minlen:
#             return False
#         d = sequence[2] - sequence[1]
#         if all(sequence[i+1] - sequence[i] == d for i in range(1, len(sequence)-1)):
#             print('弱等差数列')
#             return True



class GeometricSequenceStrategy(ValidateStrategy):
    """验证是否为等比数列"""

    name='验证_等比数列'
    def apply(self, sequence,minlen=3):
        if len(sequence) <minlen or 0 in sequence:
            return False
        q = sequence[1] / sequence[0]
        return all(sequence[i+1] / sequence[i] == q for i in range(1, len(sequence)-1))


class AddSumStrategy(ValidateStrategy):
    """验证是否为递增数列"""

    name='验证_递增数列 斐波那契数列'
    def apply(self, sequence,minlen=4):
        if len(sequence) < minlen:
            return False
        # q = sequence[1] / sequence[0]
        if sequence[2] ==0:
            return False
        return all(sequence[i+1] + sequence[i] == sequence[i+2] for i in range(0, len(sequence)-2))

class DecSumStrategy(ValidateStrategy):
    """验证是否为递增数列"""

    name='验证_递减数列'
    def apply(self, sequence,minlen=4):
        if len(sequence) < minlen:
            return False
        if sequence[2] ==0:
            return False
        # q = sequence[1] / sequence[0]
        return all(sequence[i]-sequence[i+1] == sequence[i+2] for i in range(0, len(sequence)-2))

class MutiplySumStrategy(ValidateStrategy):
    """验证是否为递乘数列"""

    name='验证_递乘数列'
    def apply(self, sequence,minlen=4):
        if len(sequence) <minlen:
            return False
        # q = sequence[1] / sequence[0]
        if sequence[2] ==0:
            return False
        return all(sequence[i]*sequence[i+1] == sequence[i+2] for i in range(0, len(sequence)-2))

class DivSumStrategy(ValidateStrategy):
    

    name='验证_递除数列'
    def apply(self, sequence,minlen=4):
        if len(sequence) <minlen:
            return False
        # q = sequence[1] / sequence[0]
        try:
            return all(sequence[i]/sequence[i+1] == sequence[i+2] for i in range(0, len(sequence)-2))
        except:
            return False


class RegressStrategy(ValidateStrategy):
    
    # 定义特征工程函数
    
    def feature_engineering(self,data,y_f):
      import numpy as np
      X = []
      Y= []
      enable_two=False
      
      if len(data)>=4:
          # enable_two=True
          pass
          
      for i, num in enumerate(data):
        # features=[]
        
        
        features = [
          i+1,    # 位置常数
          (i+1)**2,  # 位置常数的平方
          (i+1)**3,  # 位置常数的立方
          data[i-1] if i > 0 else None,  # 前一位数
          data[i+1] if i < len(data)-1 else None,  # 后一位数 
          # num**2,   # 自身的平方
          # num**3,   # 自身的立方 
        ]
        features_desc=['位置常数','位置常数的平方','位置常数的立方','前一位数','后一位数']
        if enable_two:
            features.append(data[i-2] if i > 1 else None) # 前第二位数
            # features.append(data[i+2] if i < len(data)-2 else None)# 后第二位数
        if all(x!=None for x in features):
            X.append(features)
            Y.append(num**y_f)
      return features_desc,X,Y


    def LpSolve(self,X,y):
        import numpy as np
        from pulp import LpProblem, LpMinimize, LpVariable, LpStatus,LpInteger
        import pulp

        # 设置日志级别为 0 (不输出任何信息)
        pulp.LpSolverDefault.msg = 0
        # print(X,y)
        # 输入数据 (每个子列表是一个样本)
        # X = [[3, 9, 27, 2, 7, 9, 27, 1],
        #      [4, 16, 64, 3, 46, 49, 343, 2]]
        # y = [9, 49]

        # 将 X 转换为 NumPy 数组
        X = np.array(X)

        # 创建线性规划问题
        model = LpProblem(name="linear_regression", sense=LpMinimize)

        # 定义变量 (与之前相同)
        b = LpVariable(name="b", lowBound=0, upBound=10, cat=LpInteger)
        w = [LpVariable(name=f"w_{i}", lowBound=0, upBound=10, cat=LpInteger) for i in range(X.shape[1])]
        # e = [LpVariable(name=f"e_{i}") for i in range(X.shape[0])]

        # 定义目标函数 (最小化误差平方和)
        # model += sum(e_i for e_i in e)
        # 添加约束条件 (每个样本一个约束)
        for i in range(len(X)):  # 迭代每个样本
            model += y[i] == b + sum(w[j] * X[i, j] for j in range(X.shape[1]))# + e[i]

        # model+= (b <= 10, "constraintb1")
        # model+= (b >= 10, "constraintb2")
        # for _w,i in zip(w,range(len(w))):
        #     model+= (_w <= 10, "constraint%d"%(i+1))
        #     model+= (_w >= -10, "constraintS_%d"%(i+1))

        # 求解线性规划问题
        status = model.solve()
        # 打印结果
        # print(f"状态: {LpStatus[status]}")
        # print(f"截距: {b.value()}")
        # print(f"斜率: {[w_j.value() for w_j in w]}")
        
        if LpStatus[status]=='Optimal':
            return b.value(),[w_j.value() for w_j in w]
        else:
            return None,None


    
    def apply(self, sequence,minlen=5):
        self.name='验证_回归分析（线性规划）'
        import numpy as np
        if len(sequence) <minlen:
            return False
        # q = sequence[1] / sequence[0]
        
        for i in [1,2,3]:
            features_desc,X,y = self.feature_engineering(sequence,i)
            b,ks=self.LpSolve(X,y)
            if b is None:
                continue
            mask = np.array(ks) != 0
            res=[]
            for o in range(len(mask)):
                if mask[o]:
                    res.append('(%d * %s)'%(int(ks[o]),features_desc[o]))
            
            
            name='数'
            if i>1:
                name+='的 %d 次方' %i
            name+=' = '+'+'.join(res)
            if b!=0:
                name+=' + %d' %b
            
            self.name=self.name+name
            print('方 %d'%i,sequence,b,ks)
            return True
            
            
            







class CompositeStrategy(Strategy):
    """组合策略"""

    def __init__(self, strategies):
        self.strategies = strategies

    def apply(self, sequence):
        s=sequence.copy()
        try:
            for strategy in self.strategies:
                sequence = strategy.apply(sequence)
            return sequence
        except:
            return s





class NumberSeriesSolver:
    """数字推理求解器"""

    def __init__(self, sequence, simplify_strategies, validate_strategies,validate_composite_strategies):
        self.sequence = sequence
        self.simplify_strategies = simplify_strategies
        self.validate_strategies = validate_strategies+validate_composite_strategies

    def solve(self):
        """尝试所有策略组合，找到满足验证条件的组合并得出答案"""
        sim=list(product(self.simplify_strategies,self.simplify_strategies))
        # print(sim)
        for simplify_combination in sim:
            simplified_sequence = CompositeStrategy(simplify_combination).apply(self.sequence.copy())
            
            # print([x.name for x in simplify_combination])
            # print(simplified_sequence)
            
            for validate_strategy in self.validate_strategies:
                # print(simplify_combination,validate_strategy)
                if validate_strategy.apply(simplified_sequence):
                    # 找到满足条件的策略组合，计算答案
                    
                    # print("TURE~!!")
                    print([x.name for x in simplify_combination],validate_strategy.name)
                    print(self.sequence,simplified_sequence)
                    
                    return self._calculate_answer(simplified_sequence)
        print("！！False!!",self.sequence)
        
        return None

    def _calculate_answer(self, simplified_sequence):
        """根据简化后的数列计算答案"""
        # 每个类要反向计算，先不管了，毕竟现在的解决架构也还是不太够好
        pass


# 示例用法
simplify_strategies = [
    NoneStrategy(),
    # [OddTermStrategy(), EvenTermStrategy()],
    OddEvenDecStrategy(-1),
    OddEvenDecStrategy(1),
    NeiborDivStrategy(),
    SqrtStrategy(),
    # OddTermStrategy(),
    # EvenTermStrategy(),
    # EvenOddDecStrategy(),
    GCDSimplifyStrategy(),
    DifferenceStrategy(),
    MutiplySelfStrategy(),
    PrimeFactorizationSimplifyStrategy(),
    DecXMutiplySelfStrategy(2),
    DecXMutiplySelfStrategy(3),
    DecXMutiplySelfStrategy(4),
    InsertDotStrategy(1),
    InsertDotStrategy(-1)
]

# 似乎缺点什么，如[5, 3, 10, 6, 15]拆分为双数列，双数列再同时验证
validate_strategies = [
    ArithmeticSequenceStrategy(), 
    # WeakArithmeticSequenceStrategy(),
    GeometricSequenceStrategy(),
    AddSumStrategy(),
    DecSumStrategy(),
    MutiplySumStrategy(),
    DivSumStrategy()
]
validate_composite_strategies =[
    SplitValidateStrategy(),
    RegressStrategy()   #万金油
    
    ]


seq = [
    [2,2,4,8,32,256],
    [-3,10,7,17],
    [5,3,10,6,15,12],
    [1,2,4,8],
    [1,3,2,6,5,15,14],
    [49, 64, 81, 100],
    [0, 3, 8, 15, 24],
    [2, 5, 10, 17, 26],
    [0, 6, 24, 60, 120],
    [2, 10, 30, 68],
    [12, 34, 56, 78],
    [26, 11, 31, 6, 36, 1, 41],
    [8, 12, 24, 60],
    [60, 30, 2, 15],
    [3, 3, 6, 18],
    # [1 / 343, 1 / 216, 1 / 125], #分数先不弄
    # [1, 1 / 16, 1 / 256, 1 / 625],
    [49,510,611,712], # 4,5,6,7  9,10,11,12
    [12,22,24,28], # 12得2，22得4，24得8，28十六，xx三十二      ps：这题我感觉是脑筋急转弯了，已经不算数列规律了
    # [-1, 0, 1, 2, 9], # 从第二项起后项分别是相邻前一项的立方加1，故括号内应为93+1=730。故选730  ps：解释太牵强，不管
    [1, 2, 3, 7, 46], # 本数列规律为第项自身的平方减去前一项的差等于下一项，即12-0，22-1=3，32-2=7，72-3=46，462-7=2109，故选2109
]



def solve(s):

    solver = NumberSeriesSolver(sequence=s, 
                              simplify_strategies=simplify_strategies, 
                              validate_strategies=validate_strategies,
                              validate_composite_strategies=validate_composite_strategies)
    answer = solver.solve()


for s in seq:
    solve(s)
    # print(answer)
    print('--------------')


#%%
# 滑稽结果

['平方策略', '质因数分解策略', '偶奇差数项策略 -1'] #验证_等差数列
[12, 22, 24, 28]

MutiplySelfStrategy().apply([12, 22, 24, 28])
# Out[146]: [144, 484, 576, 784]

PrimeFactorizationSimplifyStrategy().apply([144, 484, 576, 784])
# Out[147]: [12, 12, 22, 22, 24, 24, 28, 28]