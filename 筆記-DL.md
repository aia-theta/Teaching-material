
# DL

## training 正常 val/testing 壞掉
* 檢查dropout, batchnorm 是否正常設置
* 檢查 momentum/decay 設置

## BN momentum/decay （各地名稱不同）
關係到 running mean 的更新速度  
(running mean * momentum) + (batch mean * (1-momentum))  
pytorch 的 momentum 倒過來 (設0.1 = 保留0.9)  

若 momentum 太大可能導致 val/testing 時 loss 飆高/亂跳

### default value
* tensorflow
    * tf.layer 0.99
    * tf.contrib 0.999
    * tf.contrib.slim 0.999
* keras 0.99
* pytorch 0.1 (實際等同 0.9)

# 運行效能

## 跑太慢了啊啊啊啊
* 檢查 nvidia-smi
    * GPU 上有多個 process
        * 要跟別人分算力，所以慢
    * 算力沒用滿
        * feed input 的速度不夠快
            * multi thread(IO)
            * multi process(CPU)
        * 太常存 model
            * 存的間距拉大點

## 還想更快
* data_format 改用 channel_first(cuDNN)
* BN 的 fused 設為 True
* 用 Dataset API 取代 feed dict(multi GPU 較明顯)

## multi  GPU
* 做法
    * 同步
        * average grad or loss
    * 非同步
        * 分別 update

# pytorch

## load DataParallel model weight
直接改名稱(笑

```python
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in w.items():
    name = k[7:]
    new_state_dict[name] = v
```

## batchnorm 在 testing 壞掉
* 記得切換為 eval mode 
```python
model.eval()
```

## single GPU 正常, multi GPU testing 壞掉
可能是 batch 中的資料分佈不均（如前半都是 OK, 後半都是 NG）  
因為 BN 是 GPU 各做各的,會導致最後保留的 moving mean 偏向某種 class  
解決辦法: batch 內也要做 shuffle

## tensors are on different GPUs
* 檢查 model/tensor 是否都是 GPU mode
* 如有 custom function, 確認資料維持在同個 GPU 上
```python
orig_tensor.data.new(new_data)
```

# tensorflow

## batchnorm 在 testing 壞掉
可能沒有更新 moving_mean  
keras 有設 set_learning_phase 也還是要做

方案一
```python
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
```

方案二 (only for contrib/slim )
```python
tf.contrib.layers.batch_norm(x, updates_collections=None) 
```

## get variable list
```python
# all variables
tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

# all trainable variables
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
tf.trainable_variables()

# not trainable but should be update (such as moving mean in BN layer)
tf.get_collection(tf.GraphKeys.UPDATE_OPS)
```

## select parameters to optimize
```python
opt.minimize(loss, var_list=[v for tf.trainable_variables() in var_list if "CONV" in v.name])
```

## restore part of variables
```python
saver = tf.train.Saver([v for v in var_list if 'CONV' in v.name and 'biases' not in v.name])
saver.restore(sess, 'tf_ckpt/resnet_v2_50.ckpt')
```

## GAP
```python
# GAP 2D (NHWC)
x = tf.reduce_mean(x, [1,2])
```

## change variable names in ckpt
```python
scope = 'CONV_1'
ckpt = 'resnet_v1_50'

# get var list (name, shape)
vars = tf.contrib.framework.list_variables('{}.ckpt'.format(ckpt))

with tf.Graph().as_default(), tf.Session().as_default() as sess:
    new_vars = []
    for name, shape in vars:
        # get weight from ckpt by name
        v = tf.contrib.framework.load_variable('{}.ckpt'.format(ckpt), name)
        new_vars.append(tf.Variable(v, name='{}/{}'.format(scope, name)))

    saver = tf.train.Saver(new_vars)
    sess.run(tf.global_variables_initializer())
    saver.save(sess, './{}_{}.ckpt'.format(ckpt, scope))
```

## Attempting to use uninitialized value
```python
init = tf.global_variables_initializer()
sess.run(init)
```

## multi GPU
calculate grad in different GPUS and apply mean grad  
```python
tower_grads = []

with tf.variable_scope(tf.get_variable_scope()):
    for i in range(n_gpus):
        with tf.device('/gpu:{}'.format(i)):
            with tf.name_scope('TOWER_{}'.format(i)) as scope:
                x, y = reader.dequeue(batch_size)
                loss = tower_loss(x, y)

                tf.get_variable_scope().reuse_variables()

                grads = opt.compute_gradients(loss)
                tower_grads.append(grads)
                
grads = average_gradients(tower_grads)
update = opt.apply_gradients(grads)
```

## channel last/first
預設為 channel last, 可由 data_format 設定  
一般圖片開起來是 channel last, 可用 img.transpose((2, 0, 1)) 交換  

channel first 會比較快（測試起來約快一成？）  
因為符合 cuDNN 的 order  
可參考 https://www.tensorflow.org/performance/performance_guide#data_formats

## use keras in TF
keras 的 learning_phase 是相當於 TF 各 layer 的 training  
update moving mean 還是要另外綁  

### load keras pretrained model
要另外指定 keras session, 不然會把 weight load 到 keras 自己的 session
注意 init 會清掉 load 進來的 wieght  
```python
model1 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_img)
conv_output = model1.get_layer('activation_49').output
```

### You must feed a value for placeholder tensor '.../.../keras_learning_phase' with dtype bool
```python
is_training = main_graph.get_tensor_by_name('keras_learning_phase:0')
```

## TF pretrained model

小心各種不同的 preprocess

### tensorflow-slim
https://github.com/tensorflow/models/tree/master/research/slim

* Inception 1~4 & Inception-ResNet-v2
* ResNet v1/v2 50~200
* VGG 16/19
* MobileNet
* NASNet

#### 坑
* preprocess 都不一樣
* 記得確認 BN 的 decay
* 最好用自帶的 argscope
* 改成 NCHW 如果報錯，可能是因為 dimension, padding 等等是寫死的，要自己改 model

### keras

* Xception
* VGG16
* VGG19
* ResNet50
* InceptionV3
* InceptionResNetV2
* MobileNet

### https://github.com/flyyufelix/DenseNet-Keras

* DenseNet  121~169
