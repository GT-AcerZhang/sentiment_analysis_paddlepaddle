#!/usr/bin/env python
# coding: utf-8

# # 如何使用PaddleHub提供的ERNIE进行文本分类
# 
# ## 一、简介
# 
# ERNIE 通过建模海量数据中的词、实体及实体关系，学习真实世界的语义知识。相较于 BERT 学习原始语言信号，ERNIE 直接对先验语义知识单元进行建模，增强了模型语义表示能力，以 Transformer 为网络基本组件，以Masked Bi-Language Model和 Next Sentence Prediction 为训练目标，通过预训练得到通用语义表示，再结合简单的输出层，应用到下游的 NLP 任务。本示例展示利用ERNIE进行文本分类任务。
# 
# ## 二、准备工作
# 
# 请务必使用GPU环境, 因为下方的代码基于GPU环境.
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/d3d177c2dd3e47768ad8bd22fe26fc3a06b2a1ae8f274d149af72c485c5d8029)
# 
# 当前平台正在进行普遍赠送, 只要点击[此处表单](https://aistudio.baidu.com/aistudio/questionnaire?activityid=457)进行填写, 之后再度运行即可获赠. 
# 
# 首先导入必要的python包
# 

# In[1]:


# !pip install PaddlePaddle==1.5.2
get_ipython().system('pip install --upgrade paddlehub -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[2]:


#下载ernie的module
# !hub install ernie
get_ipython().system('hub install ernie_tiny')


# In[3]:


# -*- coding: utf8 -*-
import paddlehub as hub


# 接下来我们要在PaddleHub中选择ernie作为预训练模型，进行Fine-tune。ChnSentiCorp数据集是一个中文情感分类数据集。PaddleHub已支持加载该数据集。关于该数据集，详情请查看[ChnSentiCorp数据集使用](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-Dataset)。

# In[4]:


module = hub.Module(name="ernie_tiny")
dataset = hub.dataset.ChnSentiCorp()


# 如果想尝试其他语义模型（如ernie_tiny, RoBERTa等），只需要更换Module中的`name`参数即可.
# 
#    模型名                           | PaddleHub Module
# ---------------------------------- | :------:
# ERNIE, Chinese                     | `hub.Module(name='ernie')`
# ERNIE 2.0 Tiny, Chinese            | `hub.Module(name='ernie_tiny')`
# ERNIE 2.0 Base, English            | `hub.Module(name='ernie_v2_eng_base')`
# ERNIE 2.0 Large, English           | `hub.Module(name='ernie_v2_eng_large')`
# RoBERTa-Large, Chinese             | `hub.Module(name='roberta_wwm_ext_chinese_L-24_H-1024_A-16')`
# RoBERTa-Base, Chinese              | `hub.Module(name='roberta_wwm_ext_chinese_L-12_H-768_A-12')`
# BERT-Base, Uncased                 | `hub.Module(name='bert_uncased_L-12_H-768_A-12')`
# BERT-Large, Uncased                | `hub.Module(name='bert_uncased_L-24_H-1024_A-16')`
# BERT-Base, Cased                   | `hub.Module(name='bert_cased_L-12_H-768_A-12')`
# BERT-Large, Cased                  | `hub.Module(name='bert_cased_L-24_H-1024_A-16')`
# BERT-Base, Multilingual Cased      | `hub.Module(nane='bert_multi_cased_L-12_H-768_A-12')`
# BERT-Base, Chinese                 | `hub.Module(name='bert_chinese_L-12_H-768_A-12')`

# 如果想加载**自定义数据集**完成迁移学习，详细参见[自定义数据集](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E9%80%82%E9%85%8D%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E5%AE%8C%E6%88%90FineTune)

# ## 三、生成Reader
# 
# 接着生成一个文本分类的reader，reader负责将dataset的数据进行预处理，首先对文本进行切词，接着以特定格式组织并输入给模型进行训练。
# 
# `ClassifyReader`的参数有以下三个：
# * `dataset`: 传入PaddleHub Dataset;
# * `vocab_path`: 传入ERNIE/BERT模型对应的词表文件路径;
# * `max_seq_len`: ERNIE模型的最大序列长度，若序列长度不足，会通过padding方式补到`max_seq_len`, 若序列长度大于该值，则会以截断方式让序列长度为`max_seq_len`;

# In[5]:


reader = hub.reader.ClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),
    max_seq_len=128)


# **NOTE：** Reader参数max_seq_len、moduel的context接口参数max_seq_len三者应该保持一致，最大序列长度`max_seq_len`是可以调整的参数，建议值128，根据任务文本长度不同可以调整该值，但最大不超过512。

# ## 四、选择Fine-Tune优化策略
# 适用于ERNIE/BERT这类Transformer模型的迁移优化策略为`AdamWeightDecayStrategy`。详情请查看[Strategy](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-Strategy)。
# 
# `AdamWeightDecayStrategy`的参数有以下三个：
#  * `learning_rate`: 最大学习率
#  * `lr_scheduler`: 有`linear_decay`和`noam_decay`两种衰减策略可选
#  * `warmup_proprotion`: 训练预热的比例，若设置为0.1, 则会在前10%的训练step中学习率逐步提升到`learning_rate`
#  * `weight_decay`: 权重衰减，类似模型正则项策略，避免模型overfitting
#  * `optimizer_name`: 优化器名称，使用Adam

# In[6]:


strategy = hub.AdamWeightDecayStrategy(
    weight_decay=0.01,
    warmup_proportion=0.1,
    learning_rate=5e-5)


# PaddleHub提供了许多优化策略，如`AdamWeightDecayStrategy`、`ULMFiTStrategy`、`DefaultFinetuneStrategy`等，详细信息参见[策略](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-Strategy)

# ## 五、选择运行时配置
# 
# 在进行Finetune前，我们可以设置一些运行时的配置，例如如下代码中的配置，表示：
# 
# * `use_cuda`：设置为False表示使用CPU进行训练。如果您本机支持GPU，且安装的是GPU版本的PaddlePaddle，我们建议您将这个选项设置为True；
# 
# * `epoch`：要求Finetune的任务只遍历1次训练集；
# 
# * `batch_size`：每次训练的时候，给模型输入的每批数据大小为32，模型训练时能够并行处理批数据，因此batch_size越大，训练的效率越高，但是同时带来了内存的负荷，过大的batch_size可能导致内存不足而无法训练，因此选择一个合适的batch_size是很重要的一步；
# 
# * `log_interval`：每隔10 step打印一次训练日志；
# 
# * `eval_interval`：每隔50 step在验证集上进行一次性能评估；
# 
# * `checkpoint_dir`：将训练的参数和数据保存到ernie_txt_cls_turtorial_demo目录中；
# 
# * `strategy`：使用DefaultFinetuneStrategy策略进行finetune；
# 
# 更多运行配置，请查看[RunConfig](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-RunConfig)

# In[7]:


config = hub.RunConfig(
    use_cuda=True,
    num_epoch=1,
    checkpoint_dir="ernie_tiny_txt_cls_turtorial_demo",
    batch_size=100,
    eval_interval=50,
    strategy=strategy)


# ## 六、组建Finetune Task
# 
# 有了合适的预训练模型和准备要迁移的数据集后，我们开始组建一个Task。
# 
# 1. 获取module的上下文环境，包括输入和输出的变量，以及Paddle Program；
# 2. 从输出变量中找到用于情感分类的文本特征pooled_output；
# 3. 在pooled_output后面接入一个全连接层，生成Task；

# In[8]:


inputs, outputs, program = module.context(
    trainable=True, max_seq_len=128)

# Use "pooled_output" for classification tasks on an entire sentence.
pooled_output = outputs["pooled_output"]

feed_list = [
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name,
]

cls_task = hub.TextClassifierTask(
    data_reader=reader,
    feature=pooled_output,
    feed_list=feed_list,
    num_classes=dataset.num_labels,
    config=config)


# 
# 如果想改变迁移任务组网，详细参见[自定义迁移任务](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub:-%E8%87%AA%E5%AE%9A%E4%B9%89Task)

# ## 七、开始Finetune
# 
# 我们选择`finetune_and_eval`接口来进行模型训练，这个接口在finetune的过程中，会周期性的进行模型效果的评估，以便我们了解整个训练过程的性能变化。

# In[9]:


run_states = cls_task.finetune_and_eval()


# ## 八、使用模型进行预测
# 
# 当Finetune完成后，我们使用模型来进行预测，完整预测代码如下：

# In[11]:


import numpy as np

# Data to be prdicted
data = [['我不是不想说,是赖得说'],['这效率真是无语了'],['so good'],['服务态度如天使般']
    # ["这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"], ["交通方便；环境很好；服务态度很好 房间较小"],
    # [
    #     "还稍微重了点，可能是硬盘大的原故，还要再轻半斤就好了。其他要进一步验证。贴的几种膜气泡较多，用不了多久就要更换了，屏幕膜稍好点，但比没有要强多了。建议配赠几张膜让用用户自己贴。"
    # ],
    # [
    #     "前台接待太差，酒店有A B楼之分，本人check－in后，前台未告诉B楼在何处，并且B楼无明显指示；房间太小，根本不像4星级设施，下次不会再选择入住此店啦"
    # ], ["19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~"]
]

index = 0
run_states = cls_task.predict(data=data)
results = [run_state.run_results for run_state in run_states]
for batch_result in results:
    # get predict index
    batch_result = np.argmax(batch_result, axis=2)[0]
    for result in batch_result:
        print("%s\tpredict=%s" % (data[index][0], result))
        index += 1

