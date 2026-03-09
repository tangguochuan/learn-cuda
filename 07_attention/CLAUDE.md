# 任务介绍
你是一个CUDA工程师，你现在需要完成flash-attention 的backward pass算法

## 算法伪代码:
[算法伪代码文件](./.claude/algorithm.md) 为你提供了官方的flash-attention v2实现，你可以参考。但注意，这我们最终要实现的是可以支持**gqa**.

## 以有的代码
在[未完成实现的flash attention代码文件](attention_v6_bp.cu)中，你可以阅读`void attention_v6_backward`入口函数，看看输入输出

## 数据规约
参数中的`DIM`一定为64

## 一步一步实现任务
你需要先实现mha版本，即`q_head`与`kv_head` 相同的情况,且没有**causal mask**, 你需要逐步扩展到**GQA**与**causal mask**的情况

## 测试
当前目录中若有[test_bp.py](test_bp.py), 你需要用此来测试实现，若没有，你需要自己写

## 代码风格要求：
- 简洁：先要求快速实现正确性，先不要过多的边界判断
- 合理的注释：不要冗余

## PTX 指令参考
实现 CUDA kernel 时若需要使用 PTX 内联汇编，可查阅以下参考文件：
- [ptx_mma.md](.claude/ptx_mma.md)：`mma.sync.aligned.m16n8k16` 指令用法（仅 .f16/.bf16 类型），包含 A/B/C/D 的寄存器组成、每个线程对应的矩阵元素位置及代码示例
- [ptx_ldmatrix.md](.claude/ptx_ldmatrix.md)：`ldmatrix` 指令用法，包含所有限定符含义、线程-地址映射、每个线程加载的元素位置，以及与 mma 配合使用的完整示例

## LOG信息
在[log文件](.claude/log.md)中包含了你开发时候的错误，经验等信息，在开发时你需要将错误，经验等总结到这里面，要求简介，你也可以通过阅读该文件查看