# 图 1：阿基米德螺线参数及龙头盘入方向

本图用于定义坐标系、极角、极径、初始点和盘入方向，不承载数值结果分析。

运行：

```powershell
python ".\图片存放\问题1\图01_阿基米德螺线参数示意图\生成图片.py"
```

建议放在论文首次给出

\[
r=b\theta,\qquad
x=b\theta\cos\theta,\qquad
y=b\theta\sin\theta
\]

之后。

符号约定：位置函数写作向量

\[
\mathbf{P}(\theta)=
\begin{pmatrix}
x(\theta)\\ y(\theta)
\end{pmatrix},
\]

其中 \(\theta,r,b,x,y\) 均为标量。后续统一采用以下约定：

- 位置向量写作 \(\mathbf{P}(\theta)\)、\(\mathbf{P}_i(t)\)；
- 速度向量写作 \(\dot{\mathbf{P}}_i(t)\)；
- \(v_i(t)=\lVert\dot{\mathbf{P}}_i(t)\rVert\) 表示线速度大小，是标量；
- \(t,p,b,N,i,\theta_i,r_i,d_i,S,\dot{\theta}_i\) 均按标量排版。
