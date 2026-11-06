# NeuFlow v2 光流估计网络架构解析

## 1. 概述

NeuFlow v2 是一种面向边缘设备的高效率光流估计网络，采用多尺度特征提取、跨帧注意力、全局匹配和轻量级循环迭代更新的网络架构。

网络采用由粗到细（Coarse-to-Fine）的处理方式，首先在低分辨率下建立两张图像之间的全局匹配关系，得到初始光流；然后逐步将特征和光流恢复到更高分辨率，通过局部相关性查询和卷积迭代更新，不断优化光流结果；最后使用学习式上采样恢复到原始图像分辨率。

网络主要由以下几个核心模块组成：

1. **CNN Encoder（特征编码器）**：提取两张输入图像在 1/8 和 1/16 分辨率下的多尺度特征，同时生成上下文特征。
2. **Cross Attention（跨帧注意力模块）**：在 1/16 分辨率下交换两张图像的特征信息，增强不同图像之间的特征对应关系。
3. **Global Matching（全局匹配模块）**：计算两张图像特征之间的全局匹配关系，得到初始光流。
4. **Correlation（局部相关性计算模块）**：计算两张图像特征之间的局部相关性，为后续光流迭代更新提供匹配信息。
5. **Refinement Module（光流细化模块）**：通过轻量级 CNN 循环更新结构，融合相关性、上下文、当前光流和隐藏状态，逐步修正光流。
6. **Feature/Context Merge（多尺度特征融合模块）**：将 1/16 分辨率的特征和上下文信息上采样后，与 1/8 分辨率特征进行融合。
7. **Upsampling（学习式上采样模块）**：根据特征预测上采样权重，将低分辨率光流恢复到原始图像分辨率。

**输入：** 两张 RGB 图像，Shape 为 `[B, 3, H, W]`

**输出：** 光流场，Shape 为 `[B, 2, H, W]`

其中：

- B：Batch Size
- H：图像高度
- W：图像宽度
- 输出的两个通道分别表示水平光流 u 和垂直光流 v。

NeuFlow v2 默认使用以下特征维度和迭代次数：

| 参数 | 数值 |
|---|---:|
| 1/16 特征通道数 | 128 |
| 1/16 上下文通道数 | 64 |
| 1/16 迭代状态通道数 | 64 |
| 1/8 特征通道数 | 128 |
| 1/8 上下文通道数 | 64 |
| 1/8 迭代状态通道数 | 64 |
| 1/16 光流更新次数 | 1 |
| 1/8 光流更新次数 | 8 |
| 相关性查询半径 | 4 |
| 相关性金字塔层数 | 1 |

---

## 2. 整体架构流程图

NeuFlow v2 的整体处理流程如下：

```text
┌──────────────────────────────────────────────────────────────────────┐
│                       NeuFlow v2 整体网络架构                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    输入图像1 [B,3,H,W]             输入图像2 [B,3,H,W]                │
│             │                              │                         │
│             └──────────────┬───────────────┘                         │
│                            │                                         │
│                            ▼                                         │
│                    Batch Concatenation                               │
│                       [2B,3,H,W]                                     │
│                            │                                         │
│                            ▼                                         │
│                  ┌────────────────────┐                              │
│                  │    CNN Encoder     │                              │
│                  │                    │                              │
│                  │  多尺度特征提取     │                              │
│                  └─────────┬──────────┘                              │
│                            │                                         │
│               ┌────────────┴────────────┐                            │
│               ▼                         ▼                            │
│        Feature s16                 Feature s8                        │
│        Context s16                 Context s8                        │
│        [2B,192,H/16,W/16]          [2B,192,H/8,W/8]                  │
│               │                         │                            │
│               ▼                         │                            │
│        ┌────────────────┐                │                            │
│        │ Cross Attention│                │                            │
│        │ 2层注意力结构   │                │                            │
│        └───────┬────────┘                │                            │
│                │                         │                            │
│                ▼                         │                            │
│        Feature0/Feature1                 │                            │
│        Context s16                      │                            │
│                │                         │                            │
│                ▼                         │                            │
│        ┌────────────────┐                │                            │
│        │ Global Matching│                │                            │
│        │ 全局特征匹配    │                │                            │
│        └───────┬────────┘                │                            │
│                │                         │                            │
│                ▼                         │                            │
│        Initial Flow s16                 │                            │
│        [B,2,H/16,W/16]                   │                            │
│                │                         │                            │
│                ▼                         │                            │
│        ┌────────────────┐                │                            │
│        │ Correlation s16│                │                            │
│        │ 局部相关性查询  │                │                            │
│        └───────┬────────┘                │                            │
│                │                         │                            │
│                ▼                         │                            │
│        ┌────────────────┐                │                            │
│        │ Refinement s16 │                │                            │
│        │ 轻量级CNN更新   │                │                            │
│        └───────┬────────┘                │                            │
│                │                         │                            │
│           迭代1次                        │                            │
│                │                         │                            │
│                ▼                         │                            │
│        Flow s16 上采样×2                 │                            │
│                │                         │                            │
│                └────────────┬────────────┘                            │
│                             │                                         │
│                             ▼                                         │
│                    Feature/Context Merge                             │
│                    s16与s8多尺度特征融合                              │
│                             │                                         │
│                             ▼                                         │
│                       Feature s8                                     │
│                       Context s8                                     │
│                             │                                         │
│                             ▼                                         │
│                    ┌────────────────┐                                │
│                    │ Correlation s8 │                                │
│                    │ 局部相关性查询  │                                │
│                    └───────┬────────┘                                │
│                            │                                         │
│                            ▼                                         │
│                    ┌────────────────┐                                │
│                    │ Refinement s8  │                                │
│                    │ 轻量级CNN更新   │                                │
│                    └───────┬────────┘                                │
│                            │                                         │
│                        迭代8次                                        │
│                            │                                         │
│                            ▼                                         │
│                       Flow s8                                        │
│                            │                                         │
│                            ▼                                         │
│                    ┌────────────────┐                                │
│                    │ Learned        │                                │
│                    │ Upsampling     │                                │
│                    └───────┬────────┘                                │
│                            │                                         │
│                            ▼                                         │
│                    输出光流 [B,2,H,W]                                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

整体可以分为三个阶段：

- **特征提取与全局匹配阶段**：通过 CNN Encoder 提取两张图像的多尺度特征，在 1/16 分辨率下利用 Cross Attention 和 Global Matching 建立全局匹配关系，得到初始光流。
- **多尺度光流细化阶段**：先在 1/16 分辨率下进行一次光流更新，然后融合 1/16 和 1/8 分辨率特征，在 1/8 分辨率下进一步进行八次光流迭代更新。
- **光流恢复阶段**：通过学习式上采样模块，将 1/8 分辨率的光流恢复到输入图像的原始分辨率。

---

## 3. CNN Encoder（特征编码器）

CNN Encoder 负责从两张输入图像中提取多尺度特征和上下文信息，为后续的全局匹配及光流细化提供基础数据。

NeuFlow v2 使用轻量级 CNN 结构进行特征提取，同时在 1/8 和 1/16 两个尺度上建立特征分支。

### 3.1 输入数据

输入为两张 RGB 图像：

```text
Image0: [B, 3, H, W]
Image1: [B, 3, H, W]
```

在进入 Backbone 之前，首先对两张图像进行归一化：

```python
img0 /= 255.
img1 /= 255.
```

随后在 Batch 维度进行拼接：

```python
images = torch.cat([img0, img1], dim=0)
```

拼接后的 Shape：

```text
[2B, 3, H, W]
```

这里需要注意，两个图像是在 Batch 维度拼接，而不是在 Channel 维度拼接。

### 3.2 网络结构

CNN Encoder 主要由多个 ConvBlock 组成。

每个 ConvBlock 包含：

- Conv2d：卷积操作。
- BatchNorm2d：批归一化。
- LeakyReLU：非线性激活，negative_slope=0.1。
- 第二个 3×3 Conv2d：进一步提取特征。

整体结构如下：

```text
┌────────────────────────────────────────────────────────────────┐
│                         CNN Encoder                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│                  Input [2B,3,H,W]                              │
│                          │                                     │
│                          ▼                                     │
│                   Average Pooling ×2                           │
│                          │                                     │
│                    [2B,3,H/2,W/2]                              │
│                          │                                     │
│              ┌───────────┴────────────┐                        │
│              │                        │                        │
│              ▼                        ▼                        │
│        ConvBlock 8_1            Average Pooling ×2              │
│        Kernel 8×8                       │                       │
│        Stride 4                         ▼                       │
│        Output 256                 [2B,3,H/4,W/4]                │
│              │                        │                        │
│              │                        ▼                        │
│              │                  ConvBlock 8_2                  │
│              │                  Kernel 6×6                     │
│              │                  Stride 2                       │
│              │                  Output 128                     │
│              │                        │                        │
│              └───────────┬────────────┘                        │
│                          │                                     │
│                          ▼                                     │
│                    Channel Concat                              │
│                    [2B,384,H/8,W/8]                            │
│                          │                                     │
│                          ▼                                     │
│                    ConvBlock cat_8                            │
│                    Output 192                                  │
│                          │                                     │
│                          ├───────────────────────┐             │
│                          │                       │             │
│                          ▼                       ▼             │
│                    Feature s8              Context s8          │
│                    128 Channels             64 Channels        │
│                          │                                     │
│                          ▼                                     │
│                    Average Pooling ×2                           │
│                          │                                     │
│                          ▼                                     │
│                    ConvBlock 16_1                              │
│                    Kernel 6×6                                  │
│                    Stride 2                                    │
│                    Output 128                                  │
│                          │                                     │
│                          ├───────────────┐                     │
│                          │               │                     │
│                          │               ▼                     │
│                          │         ConvBlock 8_16              │
│                          │         Kernel 6×6                  │
│                          │         Stride 2                    │
│                          │         Output 128                  │
│                          │               │                     │
│                          └───────┬───────┘                     │
│                                  │                             │
│                                  ▼                             │
│                            Channel Concat                      │
│                                  │                             │
│                                  ▼                             │
│                            ConvBlock cat_16                    │
│                            Output 190                          │
│                                  │                             │
│                                  ▼                             │
│                         Position Embedding                     │
│                         2 Channels                             │
│                                  │                             │
│                                  ▼                             │
│                         Feature s16                            │
│                         128 Channels                            │
│                         Context s16                            │
│                         64 Channels                             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

其中，网络通过两条不同的特征提取路径生成 1/8 分辨率特征：

1. 第一条路径：通过 Kernel=8、Stride=4 的卷积块直接提取特征。
2. 第二条路径：进一步对输入图像进行平均池化，然后通过 Kernel=6、Stride=2 的卷积块提取特征。

两条路径得到的特征在 Channel 维度拼接，再经过 ConvBlock 融合，输出 1/8 分辨率特征。

在 1/16 分辨率下，同样使用两条路径：

1. 直接对输入图像进行下采样，再通过 ConvBlock 提取特征。
2. 将 1/8 分辨率特征经过 ConvBlock 下采样至 1/16 分辨率。

两条路径的结果拼接后，通过 ConvBlock 进行融合，并额外拼接两个位置编码通道。

### 3.3 输出特征

默认配置下，CNN Encoder 输出两组多尺度特征：

| 输出 | Shape | 说明 |
|---|---|---|
| Feature s16 + Context s16 | [2B,192,H/16,W/16] | 1/16 尺度特征 |
| Feature s8 + Context s8 | [2B,192,H/8,W/8] | 1/8 尺度特征 |

经过特征拆分后：

| 输出 | Shape | 作用 |
|---|---|---|
| Feature s16 | [2B,128,H/16,W/16] | 1/16 尺度匹配特征 |
| Context s16 | [B,64,H/16,W/16] | 1/16 尺度上下文特征 |
| Feature s8 | [2B,128,H/8,W/8] | 1/8 尺度匹配特征 |
| Context s8 | [B,64,H/8,W/8] | 1/8 尺度上下文特征 |

其中：

- Feature 用于建立两张图像之间的特征匹配关系。
- Context 用于为后续光流细化模块提供上下文信息。

需要注意：虽然两张图像在 Batch 维度进行了拼接，但是 Context 在后续拆分时只保留第一组图像对应的上下文特征，作为两张图像共同的更新上下文输入。

---

## 4. Cross Attention（跨帧注意力模块）

Cross Attention 负责在两张图像的特征之间建立全局信息交互，增强不同图像特征之间的对应关系。

该模块只作用于 1/16 分辨率特征，减少注意力计算的空间规模。

### 4.1 输入数据

输入来自 CNN Encoder：

```text
Features s16: [2B, 192, H/16, W/16]
```

这里的 192 个通道由以下两部分组成：

```text
Feature Channels: 128
Context Channels: 64
```

### 4.2 网络结构

Cross Attention 使用两层 TransformerLayer 结构。

首先将特征图转换为序列形式：

```text
[B, C, H/16, W/16]
          │
          ▼
Flatten + Permute
          │
          ▼
[B, H/16 × W/16, C]
```

然后将两张图像的特征分开，分别作为 Attention 的 Query 和 Key/Value。

```text
┌──────────────────────────────────────────────────────────────┐
│                      Cross Attention                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│           Feature0                 Feature1                   │
│               │                        │                      │
│               ▼                        ▼                      │
│             Query                  Key / Value                │
│               │                        │                      │
│               ▼                        ▼                      │
│           Linear Q                 Linear K                   │
│               │                        │                      │
│               │                   Linear V                    │
│               │                        │                      │
│               └──────────┬─────────────┘                      │
│                          │                                    │
│                          ▼                                    │
│                Scaled Dot Product Attention                   │
│                          │                                    │
│                          ▼                                    │
│                    Linear Merge                               │
│                          │                                    │
│                          ▼                                    │
│                       LayerNorm                               │
│                          │                                    │
│                          ▼                                    │
│                     Feature Fusion                            │
│                          │                                    │
│                          ▼                                    │
│                         FFN                                   │
│                          │                                    │
│                          ▼                                    │
│                       LayerNorm                               │
│                          │                                    │
│                          ▼                                    │
│                    Residual Connection                        │
│                          │                                    │
│                          ▼                                    │
│                    TransformerLayer                           │
│                          │                                    │
│                     重复2层                                    │
│                          │                                    │
│                          ▼                                    │
│                    Feature s16                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

主要计算过程：

1. 将第一张图像的特征作为 Query。
2. 将第二张图像的特征作为 Key 和 Value。
3. 通过 Scaled Dot Product Attention 计算跨图像注意力。
4. 将注意力结果经过 Linear 和 LayerNorm 处理。
5. 通过 FFN 进一步融合原始特征与注意力结果。
6. 使用残差连接保留原始特征信息。

注意力计算形式：

\[
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d}}\right)V
\]

这里采用单头注意力结构，默认执行两层 TransformerLayer。

每一层完成后，会交换两张图像的特征顺序，使下一层能够进行反方向的信息交互。

### 4.3 输出数据

输出：

```text
Features s16: [2B,192,H/16,W/16]
```

经过特征拆分：

| 输出 | Shape |
|---|---|
| Feature0 s16 | [B,128,H/16,W/16] |
| Feature1 s16 | [B,128,H/16,W/16] |
| Context s16 | [B,64,H/16,W/16] |

其中：

- Feature0 和 Feature1 用于后续全局匹配与相关性计算。
- Context s16 用于后续 1/16 分辨率光流细化。

---

## 5. Global Matching（全局匹配模块）

Global Matching 负责计算两张图像之间的全局特征匹配关系，并根据匹配结果生成初始光流。

该模块在 1/16 分辨率下执行。

### 5.1 输入数据

输入来自 Cross Attention：

```text
Feature0 s16: [B,128,H/16,W/16]
Feature1 s16: [B,128,H/16,W/16]
```

### 5.2 全局匹配计算

首先将两张特征图展开为序列：

```text
Feature0:
[B,128,H/16,W/16]
          │
          ▼
[B,N,128]

Feature1:
[B,128,H/16,W/16]
          │
          ▼
[B,N,128]
```

其中：

\[
N=\frac{H}{16}\times\frac{W}{16}
\]

网络预先生成一个二维坐标网格：

```text
Grid: [B,2,H/16,W/16]
```

并将其展开为：

```text
Flatten Grid: [B,N,2]
```

随后通过 Scaled Dot Product Attention 计算匹配位置：

```python
correspondence = F.scaled_dot_product_attention(
    feature0,
    feature1,
    flatten_grid
)
```

其含义是：

- Query：第一张图像的特征。
- Key：第二张图像的特征。
- Value：第二张图像对应位置的坐标。

通过注意力权重对第二张图像的坐标进行加权，得到第一张图像各位置在第二张图像中的对应坐标。

### 5.3 初始光流计算

得到对应坐标后，将其恢复为二维特征图：

```text
Correspondence: [B,2,H/16,W/16]
```

再减去原始坐标网格：

\[
Flow_0=Correspondence-Grid
\]

最终得到初始光流：

```text
Initial Flow:
[B,2,H/16,W/16]
```

其中两个通道分别表示水平和垂直方向的位移。

该初始光流将作为后续 1/16 分辨率 Refinement 模块的输入。

---

## 6. Correlation（局部相关性计算模块）

Correlation 负责计算两张图像特征之间的局部相关性，为后续光流细化提供特征匹配信息。

NeuFlow v2 分别在 1/16 和 1/8 分辨率下建立独立的相关性计算模块。

### 6.1 输入数据

1/16 分辨率：

```text
Feature0 s16: [B,128,H/16,W/16]
Feature1 s16: [B,128,H/16,W/16]
```

1/8 分辨率：

```text
Feature0 s8: [B,128,H/8,W/8]
Feature1 s8: [B,128,H/8,W/8]
```

### 6.2 相关性计算

首先将两张特征图展开：

```text
Feature0: [B,128,N]
Feature1: [B,128,N]
```

通过矩阵乘法计算两张图像特征之间的相关性：

\[
Corr=\frac{Feature0^T\times Feature1}{\sqrt{C}}
\]

其中：

- C：特征通道数，默认 128。
- N：特征图的空间位置数量。

得到的相关性数据经过重新组织：

```text
Corr: [B×N,1,h,w]
```

其中 h、w 为当前特征图的空间尺寸。

NeuFlow v2 默认配置：

| 参数 | 数值 | 说明 |
|---|---:|---|
| Correlation Levels | 1 | 相关性层数 |
| Radius | 4 | 局部查询半径 |
| 查询窗口 | 9×9 | 局部查询范围 |
| 每个位置的相关性通道 | 81 | 9×9 个相关性值 |

### 6.3 Correlation Lookup

在每次光流迭代时，网络根据当前光流计算对应的目标坐标：

\[
Coords=Grid+Flow
\]

然后以该坐标为中心，在相关性特征图中查询周围 9×9 区域的相关性值。

查询过程使用双线性采样（Bilinear Sampling），获得当前位置附近的匹配信息。

```text
Current Flow
     │
     ▼
Grid + Flow
     │
     ▼
Current Coordinates
     │
     ▼
9×9 Local Correlation Lookup
     │
     ▼
Bilinear Sampling
     │
     ▼
Correlation Features
[B,81,h,w]
```

这里需要注意，NeuFlow v2 默认只有一个 Correlation Level，因此相关性查询结果为 81 个通道，而不是多个金字塔层级的相关性特征拼接结果。

相关性查询结果会送入 Refinement 模块，参与后续光流更新。

---

## 7. Refinement Module（光流细化模块）

Refinement Module 是 NeuFlow v2 中负责光流迭代更新的核心模块。

该模块采用轻量级 CNN 结构，将当前光流、局部相关性、上下文特征以及迭代状态进行融合，通过多层卷积计算光流增量和新的迭代状态。

网络分别在 1/16 和 1/8 两个分辨率下使用 Refinement 模块。

### 7.1 输入数据

以 1/16 分辨率为例：

| 输入 | Shape | 说明 |
|---|---|---|
| Correlation | [B,81,H/16,W/16] | 局部相关性 |
| Context | [B,64,H/16,W/16] | 上下文特征 |
| Iter Context | [B,64,H/16,W/16] | 当前迭代状态 |
| Flow | [B,2,H/16,W/16] | 当前光流 |
| Radius Embedding | [B,1,H/16,W/16] | 查询半径信息 |

其中 Iter Context 在第一次迭代之前初始化为全零。

Radius Embedding 默认使用半径数值 4 填充。

### 7.2 网络结构

Refinement 模块首先将所有输入在 Channel 维度进行拼接：

\[
Input=Corr+Context+IterContext+Flow+Radius
\]

因此输入通道数为：

\[
81+64+64+2+1=212
\]

网络结构如下：

```text
┌──────────────────────────────────────────────────────────────┐
│                    Refinement Module                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│    Correlation [B,81,H,W]                                    │
│    Context [B,64,H,W]                                        │
│    Iter Context [B,64,H,W]                                   │
│    Current Flow [B,2,H,W]                                    │
│    Radius Embedding [B,1,H,W]                                │
│                 │                                            │
│                 ▼                                            │
│           Channel Concat                                     │
│           [B,212,H,W]                                        │
│                 │                                            │
│                 ▼                                            │
│          ┌───────────────┐                                   │
│          │ ConvBlock 1   │                                   │
│          │ Conv 3×3      │                                   │
│          │ LeakyReLU     │                                   │
│          └───────┬───────┘                                   │
│                  │                                           │
│                  ▼                                           │
│           [B,128,H,W]                                        │
│                  │                                           │
│                  ▼                                           │
│          ┌───────────────┐                                   │
│          │ ConvBlock 2   │                                   │
│          │ Conv 3×3      │                                   │
│          │ LeakyReLU     │                                   │
│          └───────┬───────┘                                   │
│                  │                                           │
│                  ▼                                           │
│           [B,Inter,H,W]                                      │
│                  │                                           │
│                  ▼                                           │
│          ┌───────────────┐                                   │
│          │ ConvBlock 3   │                                   │
│          │ Conv 3×3      │                                   │
│          │ LeakyReLU     │                                   │
│          └───────┬───────┘                                   │
│                  │                                           │
│                  ▼                                           │
│          ┌───────────────┐                                   │
│          │ ConvBlock 4   │                                   │
│          │ Conv 3×3      │                                   │
│          │ LeakyReLU     │                                   │
│          └───────┬───────┘                                   │
│                  │                                           │
│                  ▼                                           │
│             ...                                              │
│                  │                                           │
│                  ▼                                           │
│          ┌───────────────┐                                   │
│          │ ConvBlock 7   │                                   │
│          │ Conv 3×3      │                                   │
│          │ LeakyReLU     │                                   │
│          └───────┬───────┘                                   │
│                  │                                           │
│                  ▼                                           │
│              Conv 3×3                                        │
│                  │                                           │
│                  ▼                                           │
│          [B,66,H,W]                                          │
│                  │                                           │
│          ┌───────┴────────┐                                  │
│          │                │                                  │
│          ▼                ▼                                  │
│       前2个通道         后64个通道                             │
│       Delta Flow       Iter Context                          │
│          │                │                                  │
│          ▼                ▼                                  │
│       光流增量          HardTanh                              │
│          │                │                                  │
│          ▼                ▼                                  │
│       Flow更新        Updated Iter Context                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

其中 Inter 表示中间特征通道数：

- 1/16 分辨率：Inter = 128。
- 1/8 分辨率：Inter = 96。

每个 Refinement 模块的卷积结构为：

1. ConvBlock 1：输入 212 通道，输出 128 通道。
2. ConvBlock 2：输入 128 通道，输出 Inter 通道。
3. 后续连续 5 个 ConvBlock：保持 Inter 通道不变。
4. 最后通过 Conv2d 输出 66 个通道。

每个 ConvBlock 包含：

- 3×3 Conv2d。
- LeakyReLU，negative_slope=0.1。

最后一个卷积层输出：

\[
64+2=66
\]

其中前两个通道用于预测光流增量，后 64 个通道用于更新迭代状态。

### 7.3 迭代状态更新

网络将最后输出的 66 个通道拆分为两部分：

| 输出 | Shape | 作用 |
|---|---|---|
| Delta Flow | [B,2,h,w] | 当前光流增量 |
| Iter Context | [B,64,h,w] | 更新后的迭代状态 |

Iter Context 经过 HardTanh 激活函数限制数值范围：

\[
HardTanh(x)=\max(-4,\min(4,x))
\]

光流通过累加增量进行更新：

\[
Flow_{t+1}=Flow_t+\Delta Flow_t
\]

其中：

- Flow_t：当前光流。
- Delta Flow：当前迭代预测的光流增量。
- Flow_{t+1}：更新后的光流。

更新后的 Iter Context 会作为下一次迭代的输入。

需要注意，NeuFlow v2 的迭代更新采用 CNN 卷积结构生成新的迭代状态，而不是通过 GRU 或 LSTM 单元完成状态更新。

---

## 8. Multi-Scale Feature/Context Merge（多尺度特征融合模块）

完成 1/16 分辨率光流细化后，网络需要将特征和光流恢复到 1/8 分辨率，进一步进行更高分辨率的光流更新。

该阶段通过多尺度特征融合，将 1/16 分辨率的特征和上下文信息与原有的 1/8 分辨率特征进行融合。

### 8.1 输入数据

1/16 分辨率特征：

```text
Feature s16: [2B,128,H/16,W/16]
Context s16: [B,64,H/16,W/16]
Flow s16:    [B,2,H/16,W/16]
```

1/8 分辨率特征：

```text
Feature s8: [2B,128,H/8,W/8]
Context s8: [B,64,H/8,W/8]
```

### 8.2 光流上采样

首先将 1/16 分辨率光流通过最近邻插值恢复到 1/8 分辨率：

```python
flow0 = F.interpolate(
    flow0,
    scale_factor=2,
    mode='nearest'
) * 2
```

得到：

```text
Flow s8: [B,2,H/8,W/8]
```

这里乘以 2 是因为特征图空间尺寸扩大了两倍，光流的水平和垂直位移也需要同步进行尺度调整。

### 8.3 特征融合

首先将 1/16 分辨率特征通过最近邻插值上采样：

```text
Feature s16:
[2B,128,H/16,W/16]
          │
          ▼
Nearest Interpolation ×2
          │
          ▼
[2B,128,H/8,W/8]
```

然后与原有的 1/8 分辨率特征进行拼接：

```text
Feature s8:
[2B,128,H/8,W/8]

Feature s16 Upsampled:
[2B,128,H/8,W/8]
          │
          ▼
Channel Concatenation
          │
          ▼
[2B,256,H/8,W/8]
          │
          ▼
Conv 3×3 (256→128)
          │
          ▼
GELU
          │
          ▼
Conv 3×3 (128→128)
          │
          ▼
BatchNorm
          │
          ▼
Merged Feature s8
[2B,128,H/8,W/8]
```

上下文特征采用相同的融合方式：

```text
Context s8: [B,64,H/8,W/8]

Context s16 Upsampled:
[B,64,H/8,W/8]
          │
          ▼
Channel Concatenation
          │
          ▼
[B,128,H/8,W/8]
          │
          ▼
Conv 3×3 (128→64)
          │
          ▼
GELU
          │
          ▼
Conv 3×3 (64→64)
          │
          ▼
BatchNorm
          │
          ▼
Merged Context s8
[B,64,H/8,W/8]
```

### 8.4 输出数据

| 输出 | Shape | 作用 |
|---|---|---|
| Merged Feature s8 | [2B,128,H/8,W/8] | 供 1/8 尺度相关性计算使用 |
| Merged Context s8 | [B,64,H/8,W/8] | 供 1/8 尺度光流更新使用 |
| Flow s8 | [B,2,H/8,W/8] | 1/8 尺度初始光流 |

经过该模块后，网络进入 1/8 分辨率的光流细化阶段。

---

## 9. 多尺度光流迭代更新

NeuFlow v2 分别在 1/16 和 1/8 两个分辨率下执行光流迭代更新。

两个阶段使用相同的 Refinement Module 设计，但采用不同的中间特征维度。

### 9.1 1/16 分辨率迭代

首先通过 Global Matching 得到初始光流：

```text
Flow s16: [B,2,H/16,W/16]
```

然后执行：

1. 根据当前光流，在 Correlation s16 中查询局部相关性。
2. 将相关性、Context、Iter Context、Flow 和 Radius Embedding 拼接。
3. 输入 Refinement s16。
4. 预测光流增量和新的迭代状态。
5. 累加光流增量，更新当前光流。

默认执行 1 次迭代。

### 9.2 1/8 分辨率迭代

1/16 分辨率更新完成后，通过多尺度融合模块得到 1/8 分辨率特征、上下文和初始光流。

然后执行：

1. 根据当前光流，在 Correlation s8 中查询局部相关性。
2. 将相关性、Context、Iter Context、Flow 和 Radius Embedding 拼接。
3. 输入 Refinement s8。
4. 预测光流增量和新的迭代状态。
5. 累加光流增量，更新当前光流。

默认执行 8 次迭代。

### 9.3 迭代参数总结

| 阶段 | 特征分辨率 | 更新次数 | 中间特征维度 |
|---|---|---:|---:|
| Refinement s16 | 1/16 | 1 | 128 |
| Refinement s8 | 1/8 | 8 | 96 |

因此，默认推理过程中总共执行：

\[
1+8=9
\]

次光流更新。

经过 1/8 分辨率的 8 次更新后，得到最终的低分辨率光流：

```text
Flow s8: [B,2,H/8,W/8]
```

该结果将送入最后的学习式上采样模块。

---

## 10. Learned Upsampling（学习式上采样）

经过多次光流迭代后，网络得到的光流仍然处于输入图像的 1/8 分辨率。

为了恢复到原始图像尺寸，NeuFlow v2 使用学习式凸上采样（Convex Upsampling）。

该模块根据图像特征和当前光流预测局部邻域的上采样权重，再通过加权组合得到高分辨率光流。

### 10.1 输入数据

上采样模块需要两部分输入：

```text
Feature s1: [B,128,H/8,W/8]
Flow s8:    [B,2,H/8,W/8]
```

其中 Feature s1 通过额外的卷积模块从第一张原始输入图像中提取：

```text
Image0 [B,3,H,W]
       │
       ▼
ConvBlock
Kernel 8×8
Stride 8
       │
       ▼
Feature s1
[B,128,H/8,W/8]
```

### 10.2 上采样权重预测

首先将 Feature s1 和 Flow s8 在 Channel 维度拼接：

```text
Feature s1: [B,128,H/8,W/8]
Flow s8:    [B,2,H/8,W/8]
                  │
                  ▼
           Channel Concat
                  │
                  ▼
           [B,130,H/8,W/8]
```

然后通过三层卷积预测上采样权重：

```text
┌──────────────────────────────────────────────────────────────┐
│                     Upsampling Weight                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│             Feature s1 + Flow s8                              │
│                       │                                      │
│                       ▼                                      │
│                 Conv 3×3                                     │
│                 130→256                                      │
│                       │                                      │
│                       ▼                                      │
│                     ReLU                                     │
│                       │                                      │
│                       ▼                                      │
│                 Conv 3×3                                     │
│                 256→512                                      │
│                       │                                      │
│                       ▼                                      │
│                     ReLU                                     │
│                       │                                      │
│                       ▼                                      │
│                 Conv 1×1                                     │
│                 512→576                                      │
│                       │                                      │
│                       ▼                                      │
│                Upsampling Mask                               │
│                [B,576,H/8,W/8]                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

其中：

\[
576=8\times8\times9
\]

表示每个低分辨率位置需要预测 8×8 个子像素位置的上采样权重，每个子像素位置对应周围 3×3 邻域的 9 个权重。

### 10.3 凸上采样过程

预测得到的 Mask 首先进行维度变换：

```text
[B,576,h,w]
      │
      ▼
[B,1,9,8,8,h,w]
      │
      ▼
Softmax
      │
      ▼
局部9个权重归一化
```

随后通过 Unfold 操作提取当前光流周围 3×3 邻域的光流值：

```text
Flow s8: [B,2,h,w]
      │
      ▼
Unfold 3×3
      │
      ▼
[B,2,9,1,1,h,w]
```

最后使用预测的权重对 3×3 邻域光流进行加权求和：

\[
Flow_{up}=\sum_{i=1}^{9}Weight_i\times Flow_i
\]

将每个低分辨率位置对应的 8×8 子像素结果重新排列，得到：

```text
Upsampled Flow:
[B,2,8h,8w]
```

最后在网络输出处乘以 8，对光流位移进行尺度恢复：

```python
up_flow0 = self.upsample_s8(feature0_s1, flow0) * 8
```

最终输出：

```text
Final Flow: [B,2,H,W]
```

需要注意，该模块不是简单的双线性插值，而是利用网络预测的权重，对周围 3×3 邻域的光流进行加权组合，从而恢复高分辨率光流。

---

## 11. 网络参数与特征维度总结

### 11.1 核心模块汇总

| 模块 | 主要功能 | 主要输出 |
|---|---|---|
| CNN Encoder | 提取两张图像的多尺度特征 | s16、s8 特征与上下文 |
| Cross Attention | 两张图像之间的全局特征交互 | 增强后的 s16 特征 |
| Global Matching | 计算全局匹配关系 | 初始光流 s16 |
| Correlation s16 | 计算局部相关性 | 81 通道相关性特征 |
| Refinement s16 | 更新 1/16 光流 | 光流增量与迭代状态 |
| Feature/Context Merge | 融合不同尺度特征 | 融合后的 s8 特征 |
| Correlation s8 | 计算 1/8 尺度局部相关性 | 81 通道相关性特征 |
| Refinement s8 | 更新 1/8 光流 | 光流增量与迭代状态 |
| Feature s1 | 提取上采样所需特征 | 128 通道特征 |
| Learned Upsampling | 恢复光流空间分辨率 | [B,2,H,W] |

### 11.2 关键参数汇总

| 参数 | 默认值 |
|---|---:|
| Feature dim s16 | 128 |
| Context dim s16 | 64 |
| Iter Context dim s16 | 64 |
| Feature dim s8 | 128 |
| Context dim s8 | 64 |
| Iter Context dim s8 | 64 |
| Feature dim s1 | 128 |
| Cross Attention 层数 | 2 |
| Cross Attention 头数 | 1 |
| Correlation Radius | 4 |
| Correlation Levels | 1 |
| Refinement s16 中间通道 | 128 |
| Refinement s8 中间通道 | 96 |
| Refinement 卷积层数 | 5 个重复 ConvBlock |
| s16 迭代次数 | 1 |
| s8 迭代次数 | 8 |
| 最终上采样倍率 | 8 |

### 11.3 整体数据流

```text
输入两张图像
[B,3,H,W] × 2
       │
       ▼
Batch Concatenation
[2B,3,H,W]
       │
       ▼
CNN Encoder
       │
       ├───────────────────────────┐
       │                           │
       ▼                           ▼
Feature/Context s16          Feature/Context s8
       │                           │
       ▼                           │
Cross Attention                  │
       │                           │
       ▼                           │
Global Matching                  │
       │                           │
       ▼                           │
Initial Flow s16                 │
       │                           │
       ▼                           │
Correlation s16                 │
       │                           │
       ▼                           │
Refinement s16                   │
       │                           │
       ▼                           │
光流更新1次                       │
       │                           │
       ▼                           │
Flow s16 上采样×2                │
       │                           │
       └────────────┬──────────────┘
                    │
                    ▼
             Feature/Context Merge
                    │
                    ▼
               Feature s8
               Context s8
                    │
                    ▼
              Correlation s8
                    │
                    ▼
              Refinement s8
                    │
                    ▼
                光流更新
                 8次
                    │
                    ▼
                 Flow s8
                    │
                    ▼
             Learned Upsampling
                    │
                    ▼
              输出光流
             [B,2,H,W]
```

---

## 12. 关键设计特点总结

| 设计特点 | 具体说明 |
|---|---|
| 轻量级 CNN Backbone | 采用多个卷积块构建特征提取网络，生成 1/8 和 1/16 两个尺度的特征 |
| 多尺度特征提取 | 同时提取不同分辨率的特征和上下文信息 |
| Cross Attention | 在 1/16 分辨率下进行两张图像之间的特征信息交互 |
| 全局特征匹配 | 利用 Scaled Dot Product Attention 计算两张图像之间的全局对应关系 |
| 局部相关性查询 | 根据当前光流查询周围 9×9 区域的相关性信息 |
| CNN 循环更新 | 使用多层 3×3 卷积融合当前光流、相关性、上下文和迭代状态 |
| 多尺度光流细化 | 在 1/16 和 1/8 两个分辨率下分别进行光流更新 |
| 多尺度特征融合 | 将 1/16 分辨率的特征和上下文与 1/8 分辨率特征融合 |
| 学习式上采样 | 预测局部 3×3 邻域的加权系数，恢复到原始图像分辨率 |

NeuFlow v2 的整体设计思路是：先利用低分辨率特征建立全局匹配关系，得到初始光流；再通过多尺度特征融合和轻量级 CNN 迭代更新，逐步细化光流；最后通过学习式上采样得到原始分辨率的光流场。