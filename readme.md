# 環境準備
## 目標檔案架構
```
├── dataset
│     └── public
|         ├── (S1) 
|         ├── (S2)
|         ├── (S3)
|         ├── (S4)
|         └── (S5)
│     └── CV22S_Ganzin_challenge
|         ├── (HM)
|         ├── (KL)
│     └── hidden
|         ├── (S11) 
|         ├── (S22)
|         ├── (S33)
|
├── code
│     ├── dataset.py
│     ├── eval.py
│     ├── eval_draw.py
│     ├── testModel.py
│     ├── trainModel.py
│     ├── trainTestSplit.py
│     ├── Unet.py
│     ├── challenge.py
│     ├── hidden.py
│     ├── utils.py
│     ├── requirements.txt
│     ├── model_best.pth
│     └── [log_eval.txt]
│
├── [challenge_result]
│     ├── [HM]
│         ├── (01.jpg,02.jpg...)
│         └── (conf.txt)
│     └── [KL]
│         ├── (0001.jpg,0002.jpg...)
│         └── (conf.txt)
│ 
├── (hidden_result)
│     ├── (S11)
│         ├── (01,02,03...)
│             ├── (0.png,1.png...)
│             └── (conf.txt)
│     ├── (S22)
│         ├── (01,02,03...)
│             ├── (0.png,1.png...)
│             └── (conf.txt)
│     └── (S33)
│         ├── (01,02,03...)
│             ├── (0.png,1.png...)
│             └── (conf.txt)
│ 
├── model_best.th
└── readme.md
```

## 處理環境
本作業使用 venv 搭配 requirements.txt 來做環境管理。


### 環境建立
```bash=
# 進入資料夾
cd R10725035

# 建立環境
python3 -m venv cvfinal
source cvfinal/bin/activate

# 安裝套件
cd code
python3 -m pip install -r requirements.txt
```

# 程式執行
程式執行分成以下幾個部分：
1. 訓練模型
2. 繪製 S5 data 的結果（只有 S5 public dataset）
3. 繪製 hidden data 的結果
4. 繪製 challenge data 的結果

## 執行方式
```bash=
# 訓練模型，會產生 best_model.pth 在同一層資料夾

python trainModel.py

# 以下腳本會直接使用上述產生的 best_model.pth
# 如果沒有跑 trainMode.py，請執行 
cp ./../model_best.pth ./

# 繪製 S5 data 的結果（只有 S5 public dataset）
python eval_draw.py --only_s5 True --save_result True

# 繪製hidden data的結果
# 繪製出來的mask以及conf.txt會放在hidden_result裡面
python hidden.py

# 繪製challenge的結果
# 繪製出來的mask以及conf.txt會放在challenge_result裡面
python challenge.py

```

## 可能的問題
### 1. 模型是在有 gpu 的環境下訓練的，如果跑在 cpu 的環境，可能會有錯
執行：
```
python eval_draw.py --only_s5 True --save_result True
```
錯誤：
```
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
```

解法：testModel.py line 39 改成
```
model.load_state_dict(torch.load(model_path, map_location='cpu'))
```
來源：[stackoverflow](https://stackoverflow.com/questions/56369030/runtimeerror-attempting-to-deserialize-object-on-a-cuda-device)

### 2. 沒有找到檔案
錯誤：
```
[Errno 2] No such file or directory: '../dataset/public/S1/01'
```
解法：資料放對路徑


