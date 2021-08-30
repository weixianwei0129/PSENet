# PSENet 
- 这个项目主要用于进行文本检测

## Usage
### 编译后处理
- `compile.sh`
### 训练
#### 训练数据
- 在`dataset/polygon_data.py`中配置训练和测试路径;
- 在`train.sh`中修改配置路径;

#### 模型路径
- 修改`train.py`中的`checkpoint_path`字段.
- 如果需要断点训练,设置`resume`为预训练路径.
- 