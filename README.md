# CPU-playable Pytorch NLP Examples

`CPU就能跑的PytorchNLP程序`包含了从神经语言模型到Seq2seq-Transformer翻译模型的一系列从简单到复杂（实际上都挺简单）的经典自然语言处理模型。每一个模型都对应了一个自然语言处理中的实际任务。

这些项目的特点就是可以在仅需CPU上训练几分钟就能完成的同时也可以看到一些实际的有趣的结果，包括了感情分析，词性标注，词嵌入方法等（Transformer一节除外，需要在支持CUDA的平台运行）。这些项目实现的Pytorch模块均可以近似达到Pytorch原生实现的效率和表现（如果Pytorch有实现的话。结论通过比较两种模块在该项目中的任务上的训练速度和精度得出）。

项目实现的思路和实现的模型的选择参考了[nlp-tutorial](https://github.com/graykode/nlp-tutorial/)，这个项目中的大多数代码少于100行，十分精炼，而本仓库出于编写注释需要和模块化设计的角度，实现的单个项目整体大多在200行以上。除了代码行数差距以外。这个项目实现了[nlp-tutorial](https://github.com/graykode/nlp-tutorial/中实现的所有模型)，为每个模型设计了训练集，规避了原来实现中的一些代码细节问题。

该项目中的训练数据均为自动生成或者来自torchtext中的数据集。下面是该项目的七个部分的介绍：

1. **神经网络语言模型（NNLM)**  该部分位于`1.NNLM.ipynb`。根据[A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)实现了一个神经网络语言模型，得到了一个可以计算输入文本困惑度的语言模型。
2. **Word2Vec**  该部分位于`2.Word2Vec`文件夹下。实现了支持SoftMax，Hierachical SoftMax和负采样方法，CBOW和Skip-gram的Word2Vec词嵌入算法。如果加载目录里的预训练模型`2.Word2Vec/word2vec.model`，还可以通过可视化方法观察到词向量之间的聚类和平行四边形关系。
3. **TextCNN**  该部分位于`3.TextCNN.ipynb`。根据论文[Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)实现了使用卷积神经网络的文本情感分类，在IMDB数据集上达到了82%的验证集准确率。
4. **RNN**  该部分位于`4.RNN`文件夹下。根据循环神经网络公式实现了RNN，GRU，LSTM和双向LSTM，在词性标注数据集上验证比较了不同方法之间的有效性。
5. **Seq2Seq**  该部分位于`5.Seq2seq`文件夹下。实现了支持Attention的Seq2seq模型，支持配置循环神经网络类型和是否使用双向网络。在输入随机字母序列并输出反向字母序列的任务中训练，比较了不同网络类型在不同长度的序列记忆任务中的表现，还绘制了Attention权重的可视化图像。
6. **Transformer**  该部分位于 `6.Transformer`文件夹下。实现了一个Transformer，用于德语到英语的Seq2seq翻译任务。Transformer就不太可能在CPU上训练了，您可能需要自己想办把代码搬到Colab等支持CUDA的平台上运行。
7. **Bert**  该部分位于`Bert.ipynb`。实现了一个Bert模型，显然Bert已经基本上不存在能被Colab等单卡平台训练的可能性了。但是Bert的运行还是可以在CPU上完成的，这篇Notebook简单介绍演示了从Huggingface的Transformers库获取预训练的Bert模型并运行的方式。简单介绍了AlBert，RoBerta，ELECTRA，DistilBert，ERNIE（百度），GPT，CTRL，XLNet和Bart的特点以及之间的异同。

这个项目坚持PEP8的编码规范，提供了必要的注释文档和Markdown介绍，希望能给您一个愉快的阅读体验！