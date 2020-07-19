
author:young
深度学习推荐算法细节

[mlp层]         多层dense
[pooling层]     concat/sum/avg 
[embedding层]   sparse->dense


##综述
（1）DeepFM两个部分共享底层输入特征，无需进行特征工程，而Wide&Deep模型两侧保持输入独立。Wide&Deep模型将低阶与高阶特征同时建模，但是在Wide侧通常需要更多的特征工程工作。
（2）FNN模型与PNN模型将重心放在提取高阶特征信息，PNN中Product Layer精心构建低阶交叉特征信息（小于等于2阶），但是仅作为后续DNN的输入，并未将低阶特征与高阶特征并行连接。并且FNN需要进行参数预训练，模型构建时间开销较多。



DIN————
ESMM————阿里提出的multi-task学习，应对cvm不足的
TDM————召回过程的优惠，利用了层次softmax思想
EGES————阿里提出的关于图网如何学习的任务框架
DLRM————facebook剔除的模型并行和数据并行在工业界如何应用起来
MMOE————google提出的multi-task网络结构

----------------------------------------------------------------------




### Wide&Deep
https://zhuanlan.zhihu.com/p/142958834

###### 模型描述
W&D由浅层（或单层）的Wide部分神经网络和深层的Deep部分多层神经网络组成，
输出层采用softmax或sigmoid 综合Wide和Deep部分的输出。
Wide部分有利于增强模型的“记忆能力”，Deep部分有利于增强模型的“泛化能力”。

###### 问题
（1）FTRL是线性模型在线训练的主要方法，可以把FTRL当作一个**稀疏性很好，精度又不错的随机梯度下降方法**。

（2）为什么在Google的Wide&Deep模型中，要使用**带L1正则化项的FTRL作为wide部分的优化方法**？
这个问题 涉及到不同训练方法的区别联系、模型的稀疏性、特征选择和业务理解。
由于FTRL方法用的SGD更新梯度，可以来一个样本就训练一次，进而实现模型的在线更新。
**FTRL with L1非常注重模型的稀疏性，L1 FTRL会让Wide部分的大部分权重都为0，准备特征的时候 就不用准备那么多0权重的特征了，大大压缩了模型权重和特征向量的维度**。
稀疏性不见得一直是一个好东西，它会损伤模型的精度。但google商店的app推荐业务里，Wide部分采用两个id类特征的乘积，**特征向量维度过高导致“稀疏性”成为了关键的考量**。

（3）为什么在Google的Wide&Deep模型中，**使用AdaGrad作为deep部分的优化方法**？
Deep部分的输入特征包括连续值特征和embedding表示的dense特征，不存在严重的特征稀疏问题，所以可以用精度更好、更适用于深度学习训练的AdaGrad去训练。

（4）模型的泛化能力和记忆能力
“记忆能力”，可以简单理解为发现“直接的”、“暴力的”、“显然的”关联规则的能力。
Deep部分更黑盒一些，它把能想到的所有特征 扔进这个黑盒 去做函数的拟合，这样会“模糊”一些直接的因果关系，泛化成一些间接的、可能的相关性。

（5）Wide&Deep模型里Wide部分
原始onehot特征做交叉的特征

（6）Wide&Deep模型里Deep部分
连续值特征，onehot特征enbedding的dense特征


### DeepFM
https://zhuanlan.zhihu.com/p/94853056

###### 模型描述
将FM与DNN以**并行结构**组合在一起，FM侧与DNN侧 共享特征嵌入层（Embedding Layer），并在输出层 进行简单求和后 通过激活函数输出。
通过联合训练的方式 使模型达到最优.

###### 问题
（1）DeepFM模型里FM部分
DeepFM为了简单起见，去除了FM原始定义中的偏置项，仅保留了一阶项与二阶交叉项。
在实践部分，FM模块最终是将一阶项与二阶项进行简单concat。
一阶项直接使用原始onehot特征，二阶项使用embedding特征。
embedding部分参数需要学习，交叉二阶项部分不需要学习参数，一阶项部分系数需要学习。

（2）DeepFM模型里NN部分
一个样本，对embedding矩阵取某一列(dense层神经元数)
dense层和NN隐层第一层 连接后激活 active(Wx+b)


### FNN


### xDeepFM



### DIN

DIN用户的多峰兴趣
用了attention机制：pooling层
常规ctr预估是auc(给 一个正样本和一个负样本，正样本比负样本打分高的概率————auc的值)
用户维度GAUC，考虑对不同用户曝光次数带来的偏差影响

利用Activation Weight子网络，学习到candidate ID和behavior ID之间的相似程度
Linear
PReLU
BID point_wise CID[pooling]

行为序列拆分成不同通道，用卷积核对不同通道做卷积，做完卷积用attention和concat做pooling



### MMOE


































