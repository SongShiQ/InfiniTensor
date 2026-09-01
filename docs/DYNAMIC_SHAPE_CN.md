﻿# 动态 Shape 子图编译与执行

一个模型的形状不一定在导出时就定下来。批量大小随请求变化，图片尺寸随来源变化，序列长度随输入变化。这类模型导出成 ONNX 时，变化的那些维度被写成一个名字而不是一个数字，真正的尺寸要等到推理时才知道。

本文说明如何导入这样的模型、如何在同一个实例上连续换形状推理、以及部署固定形状时可以省下什么。

## 目录

- [快速开始](#快速开始)
- [动态维度是怎么表示的](#动态维度是怎么表示的)
- [运行时目标 Shape](#运行时目标-shape)
- [连续换形状](#连续换形状)
- [换形状要花多少内存](#换形状要花多少内存)
- [把维度钉住](#把维度钉住)
- [支持范围](#支持范围)
- [已知限制](#已知限制)
- [复现](#复现)

## 快速开始

编译并安装 python 前端：

```bash
make build
make install-python
```

跑演示程序，它自带三个动态 Shape 模型，不需要外部模型文件：

```bash
python examples/python/dynamic_shape_inference.py
```

每个模型会连续推理五次，每次换一组形状，并把结果与 onnxruntime 对比：

```
 batch [batch=1]: y has shape [1, 2, 3], differs from onnxruntime by at most 0.00e+00
 batch [batch=4]: y has shape [4, 2, 3], differs from onnxruntime by at most 2.98e-08
 image [batch=1, height=8, width=8]: y has shape [1, 4, 4, 4], differs from onnxruntime by at most 1.04e-07
sequence [batch=2, length=7]: y has shape [2, 7, 32], differs from onnxruntime by at most 3.58e-07
```

两侧都在 CPU 上用 float32 求同一张图，但求和的累加顺序是各自 kernel 的事，所以结果只在舍入范围内一致，而不是逐位相同。

也可以指定自己的模型和形状，`名字=尺寸` 按维度名给，单个数字给所有动态维度：

```bash
python examples/python/dynamic_shape_inference.py model.onnx batch=1 batch=8
python examples/python/dynamic_shape_inference.py model.onnx 1 4 16
```

## 动态维度是怎么表示的

ONNX 的每一个维度要么带一个数字，要么带一个名字。带名字的那些是导出时没有定下来的，导入后逐维记在张量上：

```python
from pyinfinitensor.onnx import OnnxStub
from pyinfinitensor import backend
import onnx

stub = OnnxStub(onnx.load("model.onnx"), backend.cpu_runtime())
x = stub.inputs["x"]
for axis, desc in enumerate(x.dim_descs()):
    print(axis, "dynamic" if desc.dynamic else "fixed", desc.name)
```

带数字的维度是**固定的**，不能改。给它另一个尺寸会被拒绝，而不是默默算错：

```python
stub.set_input([[1, 3, 224, 224]])   # 可以，batch 是动态的
stub.set_input([[1, 8, 224, 224]])   # 报错，通道数是模型写死的 3
```

这条拒绝是后面所有优化的依据：一个不可能改变的维度，凡是只由它推出来的东西也不会改变。

## 运行时目标 Shape

`Reshape` 的目标形状不要求在导入时就能算出来。导出器通常把它拼成一条子图——读出输入形状，取出其中几个维度，算一算，再拼回一个新形状：

```
Shape → Gather → Unsqueeze → Concat → Reshape
```

这条链在导入时不需要求值。`set_input` 给出真实形状后，形状推断会沿着它走一遍，`Reshape` 读到的是这一次的目标，而不是导入时的那个。演示程序里的 `sequence` 模型就是完整的一条：

```bash
python examples/python/dynamic_shape_inference.py sequence
```

参与这条链的算子有 `Shape`、`Gather`、`Unsqueeze`、`Squeeze`、`Slice`、`Concat`、`Identity`，以及维度上的算术 `Add`、`Sub`、`Mul`、`Div`、`Max`、`Min`、`FloorDiv`、`FloorMod`。

## 连续换形状

同一个实例可以连续换形状推理，不需要重新加载模型：

```python
for shape in ([1, 3, 8, 8], [2, 3, 8, 8], [8, 3, 12, 12], [3, 3, 4, 4], [1, 3, 8, 8]):
    stub.set_input([shape])
    stub.inputs["x"].copyin_numpy(data_of(shape))
    stub.run()
    print(stub.outputs["y"].shape())
```

`set_input` 会重新推断所有下游形状并重新规划内存。形状与上一次相同时整段跳过，因为重新排布内存比跑一遍图还贵，而一个已经被接受过的形状不需要再验一遍。

## 换形状要花多少内存

重新排布不等于重新分配。activation 的存储量停在已见过形状的最高水位上，后来一个装得下的形状直接复用，不再向 runtime 要内存；超出水位时按 1.5 倍增长。所以一段来回变化的形状序列，稳态下一次分配都不做。

```python
stub.activation_allocations()   # 累计向 runtime 要过几次
stub.activation_peak()          # 当前形状需要多少字节
stub.activation_capacity()      # 实际持有多少字节，即水位
stub.allocated_bytes()          # 连权重和堆一起，一共持有多少
```

`peak` 与 `capacity` 的差就是换来复用的那部分余量。`trim_memory()` 把余量还掉，代价是下一次换形状要重新分配。

`examples/python/dynamic_shape_benchmark.py` 把这件事量出来，同时对比「复用」与「每次推理后 trim」两种策略：

```bash
python examples/python/dynamic_shape_benchmark.py model.onnx \
    --sweep batch=1,1,1,1 height=32,128,256,32 width=32,128,256,32
```

在本机（CPU，两个自带模型）测得：

在本机（CPU，两个自带模型）测得。分配次数和持有字节是数出来的，多次运行完全一致；时间是测出来的，跨运行有约五成的漂移，所以给的是复用相对精确分配的倍率而不是绝对毫秒数：

| 模型 | 序列 | 分配次数（复用 / 精确） | 平均持有（复用 / 精确） | 每次推理 |
|---|---|---|---|---|
| 17 算子图像模型 | 32/128/256/32 循环 4 轮 | 0 / 12 | 5.00 MiB / 1.60 MiB | 快 11%–17% |
| 51 算子 attention | seq 8/16/32 循环 6 轮 | 0 / 36 | 0.07 MiB / 0.04 MiB | 快约 5%，在波动内 |

三点要说准：

- **分配次数被完全消掉**，稳态下是 0 而不是「变少」。
- **省下的时间落在排布那一段，不在算子执行上。** 图像模型上 12 轮交替试验测得复用的 `layout` 是精确分配的 0.42 倍（逐轮 0.37–0.48），而 `run` 的比值是 1.03，即执行时间不变。收益随张量大小放大：attention 的张量只有 0.07 MiB，分配成本相对 51 个算子的执行微不足道，脚本会自己标注差异落在试验波动内。
- **代价是多占内存**：图像模型平均持有 5.00 MiB 对 1.60 MiB，多三倍。全序列峰值两边一样，因为序列里含最大的那个形状，两边在那一刻都得持有它；差别在于之后还持不持着。

时间不要用单次总时长比较——总和被少数几次慢推理决定，而它们落在哪个策略上取决于谁先跑。脚本默认交替测 5 轮取中位试验，就是为了这个：单轮各测一次时，换个顺序结论会反过来。

## 把维度钉住

导出时留成动态、部署时只服务一个尺寸，是很常见的情况。这件事简化器不可能知道——它在部署存在之前就跑完了。告诉框架之后，整条只依赖这个维度的形状子图可以只算一次：

```python
stub.pin_dims("x", [0])          # 第 0 维从此固定在当前尺寸
dropped = stub.fold_shape_subgraph()
print(f"{dropped} 个算子被折叠掉了")
```

演示程序的 `--pin` 会对每个模型做这件事：

```bash
python examples/python/dynamic_shape_inference.py --pin
```

```
   image: pinned at this shape, 9 of those 9 shape operators became the same under every remaining inference; 0 remain
sequence: pinned at this shape, 7 of those 7 shape operators became the same under every remaining inference; 0 remain
```

`--fold` 则不钉住，只报告当前能折叠多少：

```bash
python examples/python/dynamic_shape_inference.py --fold
```

```
   image: the shape subgraph holds 9 operators, 0 of which are the same under every shape this model may be given and were worked out once; 9 remain to be computed per inference
sequence: the shape subgraph holds 7 operators, 0 of which are the same under every shape this model may be given and were worked out once; 7 remain to be computed per inference
```

在普通导入的模型上这个数字通常是 **0**。这不是缺陷：前端会先对模型做一遍简化，而简化器已经把「由 ONNX 声明的形状能推出来的部分」折完了。两者的知识来源相同，先跑的那个把它拿走。折叠能做的是简化器拿不到的那部分——部署钉住的维度。

钉住之后那个维度不能再改：

```python
stub.pin_dims("x", [0])
stub.set_input([[2, 3, 8, 8]])   # 报错，第 0 维已经钉在 1 上
```

这条拒绝是必须的。折叠的结果被写死进图里，钉住的维度事后改变会留下一个过期的答案，而过期的答案不会报错，只会算错。

## 支持范围

| | 支持情况 |
|---|---|
| 后端 | CPU（`backend.cpu_runtime()`）|
| 动态维度数量 | 不限，可以多个维度同时独立变化 |
| `Reshape` 目标 | 静态属性，或运行时张量 |
| `-1` 与 `0` 占位符 | 支持 |
| 形状子图算子 | `Shape` `Gather` `Unsqueeze` `Squeeze` `Slice` `Concat` `Identity`，及维度算术 |

## 已知限制

**形状变化只能从图的输入进入。** 换形状的唯一入口是 `set_input`。

**固定性的传播是按张量而不是按维度的。** 一个会变的维度会让它下游的每一个张量的每一个维度都被当作可变，包括明显不可能变的那些：批量大小会变的图片模型，通道数其实是固定的，但经过一层卷积之后就被描述成可变的了。缺的信息是「每个输入维度到了输出的哪一维」，只有算子本身能回答。所以读取计算结果的形状链，要等它依赖的所有维度都被钉住才会固定——这就是为什么图片模型是九个全折或者一个都不折。这个方向是安全的：把固定的说成可变，损失的只是一次本可以发生的折叠；反过来会把还需要计算的东西折掉。

**符号维度参与 `Reshape` 目标的算术时无法导入。** 导入期用「每个动态维当作 1」求解 `Reshape` 目标，所以目标里含有对符号维的整除时，占位算术不成立，导入就会失败。例如输入声明为 `["batch", "length", "width"]`、目标是 `[batch, length, 4, width/4]` 时，`width` 占位为 1，`1/4 = 0`。把这个维度在 ONNX 里声明成数字可以绕过，这也是导出时通常的做法。

**固定性的传播需要一遍形状推断。** 一个张量要先有生产者，才能读到它从生产者那里继承了什么，所以这是一遍图上的 pass 而不是在每个算子加入时完成的。导入本身不跑形状推断，所以传播在第一次 `set_input` 或 `pin_dims` 时才生效。直接用 `addOp` 搭图时需要显式调用 `shape_infer()`。

## 复现

演示程序：

```bash
python examples/python/dynamic_shape_inference.py          # 三个模型各五组形状
python examples/python/dynamic_shape_inference.py --fold   # 报告可折叠的算子数
python examples/python/dynamic_shape_inference.py --pin    # 钉住维度后折叠
```

与 onnxruntime 的一致性测试，`rtol=1e-4`、`atol=1e-5`，逐个形状比对输出形状和数值：

```bash
python -m pytest pyinfinitensor/tests/test_dynamic_shape_ort.py -v
```

误差限的依据：两侧都在 CPU 上用 float32 求同一张图，但求法不同——求和的累加顺序是各自 kernel 的事——所以结果只在舍入范围内一致，而不是逐位相同。实测的不一致比这个限低约四个数量级，其中一个测试直接断言这个余量，以免这个限松到掩盖真实缺陷。

后端测试：

```bash
build/Release/test_shape_fold          # 形状值传播与折叠
build/Release/test_nativecpu_slice     # CPU slice kernel
build/Release/test_nativecpu_matmul    # CPU matmul kernel，含 bias 与 batch
build/Release/test_graph               # 含容量复用与分配计数的校准
```

内存与延迟测量，同时验证两种策略输出逐位一致：

```bash
python examples/python/dynamic_shape_benchmark.py model.onnx --sweep batch=1,2,4,8
```

需要 `onnxruntime` 才能跑一致性对比，没有装的话相关测试会跳过而不是失败：

```bash
pip install onnxruntime
```
