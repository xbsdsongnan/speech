1)
data/local/lm/lm_tglarge.arpa.gz这个文件是链接到s5/corpus/3-gram.arpa.gz
G.fst用于对语言模型进行编码。当使用统计语言模型时，用srilm训练出来的语言模型为arpa格式，可以用arpa2fst将arpa转换成fst


使用命令：
arpa2fst --disambig-symbol=#0 --read-symbol-table=words.txt lm_tglarge.arpa G.fst


此外
fstdraw G.fst  fst.dot　##很大
dot -Tjpg fst.dot  fst.jpg　##时间很长
fstisstochastic G.fst ##这是一个诊断步骤，他打印出两个数字，最小权重和最大权重，以告诉用户FST不随机的程度
第一个数字很小，它证实没有状态的弧的概率加上最终状态明显小于1。第二个数字是重要的，这意味着有些状态具有“太多”的概率。对于具有回退的语言模型的FST来说，有一些具有“太多”概率的状态是正常的。
FST中权重的数值通常可以解释为负对数概率


2)
用 tri1 模型和 tgsmall 构建的 HCLG 图解码 dev_clean_2 集合的“1272-135031-0009”句,输出 Lattice 和 CompactLattice 的文本格式
解码网络使用 HCLG.fst 的方式， 它由 4 个 fst 经过一系列算法组合而成。分别是 H.fst、C.fst、L.fst 和 G.fst 4 个 fst 文件
G：语言模型，输入输出类型相同，实际是一个WFSA（acceptor接受机），为了方便与其它三个WFST进行操作，将其视为一个输入输出相同的WFST。
L：发音词典，输入：monophone，输出：词;
C:上下文相关，输入：triphone（上下文相关），输出：monophnoe;
H:HMM声学模型，输入：HMM transitions-ids (对 pdf-id 和 其它信息编码后的 id)，输出：triphone。


读取 Lattice 和 CompactLattice 均可用 SequentialLatticeReader，两者之间转换可使用 ConvertLattice()．

利用此步可单独得到.ali文件
compile-train-graphs tree 35.mdl L.fst ark:1272-135031-0009.tra   ark:1272-135031-0009-graphs.fsts
gmm-align-compiled   --transition-scale=1.0 --acoustic-scale=0.1 --self-loop-
scale=0.1   --beam=8 --retry-beam=189    35.mdl ark:1272-135031-0009-graphs.fsts    "ark:add-deltas --print-args=false scp:1272-135031-0009.scp ark:- |"    ark:1272-135031-0009.ali


读取特征1272-135031-0009
head -10 mfcc/raw_mfcc_dev_clean_2.1.scp | tail -1 | copy-feats scp:- ark,t:- | head

利用此步骤可单独获得晶格（lat）
  gmm-latgen-simple --beam=13.0 --acoustic-scale=0.0625 /home/ln/kaldi-master/egs/mini_librispeech/s5/exp/tri1/35.mdl \
     /home/ln/kaldi-master/egs/mini_librispeech/s5/exp/tri1/graph_nosp_tgsmall/HCLG.fst ark:1272-135031-0009_feats.ark "ark,t:|gzip -c > lat.1272-135031-0009.gz"
法二
gmm-latgen-faster --max-active=700 --beam=13 --lattice-beam=6 --acoustic-scale=0.1 --allow-partial=true --word-symbol-table=words.txt final.mdl HCLG.fst ark:delta.ark ark:lattice.ark


使用命令获得lattice词图
./utils/show_lattice.sh 1272-135031-0009   /home/ln/kaldi-master/egs/mini_librispeech/s5/exp/tri1/decode_nosp_tgsmall_dev_clean_2/lat.1.gz /home/ln/kaldi-master/egs/mini_librispeech/s5/data/lang_nosp_test_tgsmall/words.txt

CompactLattice的文本格式
lattice-to-nbest --acoustic-scale=0.1 --n=10 "ark:gunzip -c /home/ln/kaldi-master/egs/mini_librispeech/s5/exp/tri1/decode_nosp_tgsmall_dev_clean_2/lat.1272-135031-0009.gz |" ark,t:lat.1272-135031-0009.gz.txt



3)语言模型重打分 ，先减去"旧"LM概率，然后添加新的LM概率

原始：
compute-wer --text --mode=present ark:exp/tri1/decode_nosp_tgsmall_dev_clean_2/scoring/test_filt.txt ark,p:- 
%WER 34.25 [ 6897 / 20138, 976 ins, 635 del, 5286 sub ]
%SER 93.02 [ 1013 / 1089 ]
Scored 1089 sentences, 0 not present in hyp.
新：
compute-wer --text --mode=present ark:exp/tri1/decode_nosp_tgsmall_dev_clean_2/scoring/test_filt.txt ark,p:- 
%WER 27.60 [ 5559 / 20138, 891 ins, 432 del, 4236 sub ]
%SER 88.61 [ 965 / 1089 ]
Scored 1089 sentences, 0 not present in hyp.






















