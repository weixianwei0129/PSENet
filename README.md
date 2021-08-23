# PSENet 
- 这个项目主要用于进行文本检测

## Usage
### 编译后处理
- `compile.sh`
### 训练
#### 训练数据
- 在`dataset/psenet`添加自己的数据读入方式,例如新建`psenet_custom.py`文件;
  - 在其中修改数据路径;
- 在`dataset/psenet/__init__.py`中注册数据读取方法;
- 在`dataset/__init__.py`文件中的`__all__`中添加自己的方法;
- 在`config/psenet`中新建自己的配置文件,例如新建`psenet_r50_custom_736.py`文件,其中736指的是模型输入图片的高度;
- 在配置文件中的`data`字段中修改自己的数据读入方法名称;
- 在`train.sh`中修改配置路径;

#### 模型路径
- 修改`train.py`中的`checkpoint_path`字段.
- 如果需要断点训练,设置`resume`为预训练路径.
- 