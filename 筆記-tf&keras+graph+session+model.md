
# Graph

完整念請參考[此處](https://www.tensorflow.org/programmers_guide/graphs)
* 建立 Operation
    * tf.constant()
        * 建立產生 constant value 的 Operation
        * 返回 Tensor
    * tf.pow()
        * 可吃進 Tensor 進行計算 (Operation)
        * 返回 Tensor
* 建立 Tensor/Tensor like
    * Tensor
        * 有些操作會返回 Tensor（如上方舉例）
        * 直接建立
            * tf.convert_to_tensor()
    * Tensor like
        * Variable()
            * 阿...他會建立 Operation 存取 session 裡的值，然後返回個可以像 0Tensor 一樣操作的東西
            * 所以跑之前要先 init, 不然存取不到值
* 建立計算流程(flow)

```python
# 建立產生 42 的 Operation
# 返回 Tensor
a = tf.constant(42)
b = tf.constant(5)

# 建立對 a,b 進行計算的 Operation
# 返回 Tensor
c= tf.pow(a, b)
```
![image.png](attachment:image.png)

## default graph
* tf 會自己宣告一個 default graph, 沒指定就是在 default graph 上
* default graph 可用 tf.reset_default_graph() 清除重建

## 宣告/使用 Graph

```python
with tf.Graph().as_default():
    pass

g = tf.Graph()
with g.as_default():
    pass
```

## reset default Graph

```python
tf.reset_default_graph()
```

# session

* 管理計算環境
* 實際管理 Variable 的值
    * saver 是 restore 到 session
    * 要 init, Variable 裡面才會實際有值

## 宣告 session

```python
with tf.Session() as sess:
    pass

sess = tf.Session(graph=g)
```

## fetch

sess.run() 可以 fetch Tensor 或是 Operation  
會運行往回有拉到的部分 (subgraph)  
並返回結果  

![image.png](attachment:image.png)

# keras （TF backend）

keras會自己建立 graph 和 session  
可用 keras.backend.clear_session() 清除重設

## 指定 session
常用於 TF 混合 keras  
或是設定 session config

```python
sess = tf.Session()
tf.keras.backend.set_session(sess)
```

## load model 在做啥？

在當前 graph 建立 Operation 和 Tensor  
並把 weight load 到當前的 keras session  


```python

```
