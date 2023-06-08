# Easy-Langchain-LKB

The simplest Local Knowledge Base example based on Langchain and Chat-GLM

# 起因

之前langchain-ChatGLM比较火，我也尝试过使用它建立本地知识库。但是这个知识库往往进行不了几轮对话就爆显存了（4070 12G，模型int8优化）。

我想寻找原因，但这个项目的代码风格实在一言难尽。无奈只能自己研究，分析代码，并且写了一个仅包含必要代码的最小实现，以便排除干扰进行性能调优，并且供有同样爱好的同志们参考。

# 相关文章

TODO: 我会在B乎上写一篇文章介绍其原理和里面的一些细节，待填坑。

# 使用

安装requirement.txt中的库，直接运行即可。

默认文档为根目录下的document.txt，向量数据库保存在vector_store目录中。

默认依赖HuggingFace的'GanymedeNil/text2vec-large-chinese'及'chatglm-6b-int8'模型，可以自动下载，也可以使用本地的。

上述的参数都可以自行配置。代码很简单无需赘述。
