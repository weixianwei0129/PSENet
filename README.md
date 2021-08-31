# PSENet 
- 这个项目主要用于进行文本检测

## Usage
### 编译后处理
- `compile.sh`
### 配置

#### `config/pse_*.yaml`
- `model`中主要为模型配置,使用`backbone`+`neck`+`detection_head`方式;
- `loss`为模型loss计算的配置,单独构成一个`PSENet_Loss`层:
  - `loss_weights`为各个loss的权重;
- `train`为训练参数:
  - `schedule`表示在学习率衰减的epoch阶段;
  - `batch_size`训练时使用的batch_size;
- `data` 为数据加载器的参数:
  - `batch_size` 表示在数据加载器输出的数据批数量,当其大于train.batch_size时,训练时将采用当前值;
  - `short_size` 图片训练时使用(short_size, short_size)的尺寸, 测试时使用短边short_size的等比尺度;
  - `min_scale` 在做shrink时的坍缩因子;
  - `use_mosaic` 是否使用mosaic数据增强: 会产生`[0,10]`的随机数,当小于该值是会产生mosaic增强的训练数据;
- `evaluation` 表示测试时的参数

#### `train.py`
- 在`dataset/polygon.py`中配置训练/测试数据的路径(todo: 改到配置文件/参数中);
  - 数据格式如下,其中标注内容为多边形点的像素坐标;
    ```
    - img1.jpg
    - img1.txt
        x1,y1,x2,y2,...,xn,yn,word1
        x1,y1,x2,y2,...,xn,yn,word2
    ```
- run `sh train.sh`