# speech
speech-learning
第二部分：
给定一段音频，请提取12维MFCC特征和23维FBank，阅读代码预加重、分帧、加窗部分，完善作业代码中FBank特征提取和MFCC特征提取部分，并给出最终的FBank特征和MFCC特征，存储在纯文本中，用默认的配置参数，无需进行修改。
代码文件说明

代码依赖

python3
librosa

如果需要观察特征频谱，请确保自己有matplotlib依赖并将代码中相关注解解掉

注：不要修改文件默认输出test.fbank test.mfcc的文件名
文件路径说明

mfcc.py 作业代码
test.wav 测试音频
第三部分：
本次实验所用的数据为0-9（其中0的标签为Z（Zero））和o这11个字符的英文录音，每个录音的原始录音文件和39维的MFCC特征都已经提供， 实验中，每个字符用一个GMM来建模，在测试阶段，对于某句话，对数似然最大的模型对应的字符为当前语音数据的预测的标签（target）

训练数据：330句话，每个字符30句话，11个字符

测试数据：110句话，每个字符10句话，11个字符

digit_test/digit_train里面包含了测试和训练用数据，包括：

    wav.scp, 句子id到wav的路径的映射，所用到的数据wav文件的相对路径

    feats.scp, 语音识别工具kaldi提取的特征文件之一，句子id到特征数据真实路径和位置的映射

    feats.ark, 语音识别工具kaldi提取的特征文件之一，特征实际存储在ark文件中，二进制

    text, 句子id到标签的映射，本实验中标签（语音对应的文本）只能是0-9，o这11个字符

程序：

kaldi_io.py提供了读取kaldi特征的功能

utils.py 提供了一个特征读取工具

gmm_estimatior.py 核心代码，提供了GMM训练和测试的代码，需要自己完成GMM类中em_estimator 和calc_log_likelihood函数
输出：

程序最终输出一个acc.txt文件，里面记录了识别准确率
第四部分：
请用Python编程实现前向算法和后向算法,分别计算	P O λ ;
• 请用Python编程实现Viterbi算法,求最优状态序列,即最优路径	I ∗ = i 3∗ , i 4∗ , i n∗ .
第五部分：
基于GMM-HMM的语音识别系统
要求和注意事项

    认真读lab2.pdf, 思考lab2.txt中的问题
    理解数据文件
    ref文件作为参考输出，用diff命令检查自己的实现得到的输出和ref是否完全一致
    实验中实际用的GMM其实都是单高斯
    阅读util.h里面的注释，Graph的注释有如何遍历graph中state上所有的arc的方法。
    完成代码
        lab2_vit.C中一处代码
        gmm_util.C中两处代码
        lab2_fb.C中两处代码

作业说明
安装

该作业依赖g++, boost库和make命令，按如下方式安装：

    MAC: brew install boost (MAC下g++/make已内置）
    Linux(Ubuntu): sudo apt-get install make g++ libboost-all-dev
    Windows: 请自行查阅如何安装作业环境。

编译

对以下三个问题，均使用该方法编译。

make -C src

p1

    内容：完成lab2_vit.C中的用viterbi解码代码.
    运行:
        ./lab2_p1a.sh
        ./lab2_p1b.sh
    比较结果: 比较你的程序运行结果p1a.chart和参考结果p1a.chart.ref，可以使用vimdiff p1a.chart p1a.chart.ref进行比较，浮点数值差在一定范围内即可。

p2

    内容：估计模型参数,不使用前向后向算法计算统计量，而是用viterbi解码得到的最优的一条序列来计算统计量，叫做viterbi-EM. 给定align（viterbi解码的最优状态(或边）序列)，原始语音和GMM的初始值，更新GMM参数。完成src/gmm_util.C中两处代码。
    运行：./lab2_p2a.sh
    比较结果： 如p1，比较p2a.gmm p2a.gmm.ref

p3

    用前向后向算法来估计参数，完成src/lab2_fb.C中的两处代码。
    运行：
        ./lab2_p3a.sh: 1条数据，1轮迭代
        ./lab2_p3b.sh: 22条数据，1轮迭代
        ./lab2_p3c.sh: 22条数据，20轮迭代
        ./lab2_p3d.sh: 使用p3c的训练的模型，使用viterbi算法解码，结果应该和p1b的结果一样一样
    比较结果: 如p1，分别比较p3a_chart.dat/p3a_chart.ref和p3b.gmm/p3b.gmm.ref。

第六部分：
基于DNN-HMM的语音识别系统
本次共有2个，分别如下所示。
1
数据说明

本次实验所用的数据为0-9（其中0的标签为Z（Zero））和O这11个字符的英文录音所提取的39维的MFCC特征。其中

    训练数据：330句话，11个字符，每个字符30句话，训练数据位于train目录下。
    测试数据：110句话，11个字符，每个字符10句话，测试数据位于test目录下。

train/test目录下各有3个文件，分别如下：

    text: 标注文件，每一行第一列为句子id，第二列为标注。
    feats.scp: 特征索引文件，每一行第一列为句子id，第二列为特征的索引表示。
    feats.ark: 特征实际存储文件，该文件为二进制文件。

实验内容

本实验实现了一个简单的DNN的框架，使用DNN进行11个数字的训练和识别。 实验中使用以上所述的训练和测试数据分别对该DNN进行训练和测试。 请阅读dnn.py中的代码，理解该DNN框架，完善ReLU激活函数和FullyConnect全连接层的前向后向算法。 可以参考Softmax的前向和后向实现。dnn.py中代码插入位置为。

# BEGIN_LAB
# write your code here
# END_LAB

运行和检查

使用如下命令运行该实验，该程序末尾会打印出在测试集上的准确率。假设实现正确，应该得到95%以上的准确率，作者的实现分类准确率为98.18%。

python dnn.py

拓展

除了跑默认参数之外，读者还可以自己尝试调节一些超参数，并观察这些超参数对最终准确率的影响。如

    学习率
    隐层结点数
    隐层层数

读者还可以基于该框架实现神经网络中的一些基本算法，如：

    sigmoid和tanh激活函数
    dropout
    L2 regularization
    optimizer(Momentum/Adam)
    ...

实现后读者可以在该数字识别任务上应用这些算法，并观察对识别率的影响。

通过调节这些超参数和实现其他的一些基本算法，读者可以进一步认识和理解神经网络。
2

基于Kaldi理解基于DNN-HMM的语音识别系统。请安装kaldi，并运行kaldi下的标准数据集THCHS30的实验，该实验如链接所示， https://github.com/kaldi-asr/kaldi/blob/master/egs/thchs30/s5/run.sh。

THCHS30是清华大学开源的一个中文数据集，总共30小时。请基于该数据集，基于kaldi下该数据集的标注脚本，梳理基于DNN-HMM的语音识别系统的流程，其有哪些步骤，每一步的输入、输出，步骤间的相互关系等，可以把自己的理解流程化、图形化、文字化的记录下来，写下来。
第七部分：
说明

请根据"实验指导书.pdf":part2.2和part4分别完成N-gram计数和Witten-Bell算法的编写。

编译文件：Makefile
提供的C++文件介绍：

    main.C：入口函数
    util.{H,C}:提供命令行解析，读取和写出数据等功能,不必仔细阅读，可以掠过。
    lang_model.{H,C}:LM类定义，本实验主要部分内容，需要完成.C文件中count_sentence_ngrams()和get_prob_witten_bell()函数。
    lab3_lm.{H,C}:语言模型实验的wrapper函数。

数据文件：

字典：lab3.syms
训练集：minitrain.txt和minitrain2.txt
测试集：test1.txt和test2.txt

bash文件：

lab3_p1{a,b}.sh:测试N-gram计数
lab3_p3	{a,b}.sh：测试Witten-Bell smoothing算法
第八部分：
参考 openfst 官网,
1) 将图示 a)和 b)两个 WFST 写成 text 格式 a.txt.fst 和 b.txt.fst。
2) 定义 input label 和 output label 的字符表(即字符到数值的映射)。
3) 生成 a)和 b)对应的 binary 格式 WFST,记为 a.fst 和 b.fst
4) 进行 compse,并输出 out.fst
5) 打印输出的样子。
2:
从 kaldi/src/decoder/lattice-faster-decoder.cc 中查找
1) histogram pruning 的代码段
2) beam pruning 的代码段
3:
运行 kaldi/egs/mini_librispeech 至少训练完 3 音素模型 tri1.
1)
此时你的 data/lang_nosp_test_tglarge 中无 G.fst 文件,
将 data/local/lm/lm_tglarge.arpa.gz
转化为 G.fst 存于其中。[提交你的完整命令]
2)用 tri1 模型和 tgsmall 构建的 HCLG 图解码 dev_clean_2 集合的“1272-135031-0009”句,输
出 Lattice 和 CompactLattice 的文本格式。[提交你的完整命令和输出文件]
3)使用 1)中生成的 tglarge 的 G.fst 和 steps/lmrescore.sh 对
exp/tri1/decode_nosp_tgsmall_dev_clean_2 中的 lattice 重打分,汇报 wer。
