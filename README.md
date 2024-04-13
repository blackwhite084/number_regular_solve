<h1>用于找出数列规律，适用与考公考编那些题目</h1> 
例如题目：
13、乘法规律。之一：普通常规式：前两项之积等于第三项。
    [例15] 3，4，12，48，（     ）  A、96  B、36  C、192  D、576
     [解析]  这是一道典型的乘法规律题，仔细观察，前两项之积等于第三项。故选D。
    之二：乘法规律的变式：


执行代码：

```python
solve([3,4,12,48])
输出：
['', ''] 验证_递乘数列
[3, 4, 12, 48] [3, 4, 12, 48]
```



 [例21]  2，5，10，17，26，（    ）    A、43  B、34  C、35  D、37
[解析] 
  这个数是一个二级等差数列，相邻两项的差是一个公差为2的等差数列，括号内的数是26=11=37。如将所给的数列分别减1，可得1，4，9，16，25，即12，22，32，42，52，故括号内的数应为62+1=37，，其实就是n2+n。故选D。



执行代码：

```python
solve([2,5,10,17,26])
输出：
方 1 [2, 5, 10, 17, 26] 1.0 [0.0, 1.0, 0.0, 0.0, 0.0]
['', ''] 验证_回归分析（线性规划）数 = (1 * 位置常数的平方) + 1
[2, 5, 10, 17, 26] [2, 5, 10, 17, 26]
```
表示如第3个数10等于3的平方+1


