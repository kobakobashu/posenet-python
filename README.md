# posenet-python-gesture

"posenet-python-gesture"は[posenet-python](https://github.com/rwightman/posenet-python)に追加することで、ジェスチャの認識を行います。
 

# DEMO
ボディービルダーのポーズを読み込んでテストを行いました。

![ボディービルダー](https://user-images.githubusercontent.com/43237898/84505388-9763fa00-acf8-11ea-8dd5-cef46d328851.gif)

# Features
 
好きな画像に対してジェスチャ認識ができます。
 
# Requirement

[posenet-python](https://github.com/rwightman/posenet-python)に準拠します。

Tensorflowは1.x.xを使う必要があります。

私はOS X: 10.15.1、Python: 3.7.5で開発・テストを行いました。

> A suitable Python 3.x environment with a recent version of Tensorflow is required.

> Development and testing was done with Conda Python 3.6.8 and Tensorflow 1.12.0 on Linux.

> Windows 10 with the latest (as of 2019-01-19) 64-bit Python 3.7 Anaconda installer was also tested.

> If you want to use the webcam demo, a pip version of opencv (pip install opencv-python) is required instead of the conda version. Anaconda's default opencv does not include ffpmeg/VideoCapture support. Also, you may have to force install version 3.4.x as 4.x has a broken drawKeypoints binding.
 
> A conda environment setup as below should suffice:

```bash
conda install tensorflow-gpu scipy pyyaml python=3.6
pip install opencv-python==3.4.5.20
```

 
# Usage
 
1. [posenet-python](https://github.com/rwightman/posenet-python)のダウンロード

```
git clone https://github.com/rwightman/posenet-python.git
```
2. [posenet-python-gesture](https://github.com/besuboiu/posenet-python-gesture)のダウンロード

```
git clone https://github.com/rwightman/posenet-python.git
```

3. "posenet-python-gesture"の中身を全て、"posenet-python"ヘコピーする

```
cp -pR posenet-python-gesture/* posenet-python/
```

4. "posenet-python-gesture"へ移動してimagesファイルを作る

```
cd posenet-python
mkdir images
```
5. 認識したいポーズ（全身）の人物画像ファイルをimagesに移動

6. csvファイルの作成

```
python3 make_csv.py
```

7. demoの実行

```
python3 match_demo.py
```


 
# Author
 
This port and my work is in no way related to Google.
 
* 作成者: besuboiu
* E-mail: n.n.n.h.h.b.b.26@gmail.com
 
# License

"posenet-python-gesture" is under Apache License.
 
