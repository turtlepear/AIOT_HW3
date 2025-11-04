# AIOT_HW3 — SMS 垃圾郵件classifier

# 1️⃣ 前處理（Preprocessing）

本專案的前處理階段，主要對原始簡訊資料集進行清理與標準化，方便後續模型訓練。

## 步驟說明

| 步驟         | 說明 |
|--------------|------|
| 資料來源      | 原始簡訊資料集（CSV 格式） |
| 文字清理      | 移除特殊符號、統一大小寫 |
| 敏感資訊替換  | - URL → `<URL>` <br> - Email → `<EMAIL>` <br> - 電話 → `<PHONE>` <br> - 數字 → `<NUM>` |
| 標籤轉換      | 將標籤欄位轉為整數（0/1） |
| 輸出          | 清理後的 CSV：`sms_spam_clean.csv` |

## 使用範例

執行前處理程式：

```bash
python preprocessing.py --input_csv sms_spam_no_header.csv --output_csv sms_spam_clean.csv
```


# 2️ 訓練與推論（Train & Predict）

本階段使用清理後的簡訊資料集，透過 TF-IDF 向量化與邏輯迴歸（Logistic Regression）進行訓練與預測。

## 步驟說明

| 步驟             | 說明 |
|------------------|------|
| 載入資料         | 使用 `pandas.read_csv` 載入清理後 CSV |
| 特徵向量化       | 使用 `TfidfVectorizer` 將文字轉換為向量，設定最大詞彙數（`max_features`） |
| 訓練集與測試集分割 | 使用 `train_test_split` 分割資料，預設測試集比例 `0.2`，並支援 stratify 分層抽樣 |
| 模型訓練         | 使用 `LogisticRegression` 進行模型訓練，設定最大迭代次數 `max_iter=1000` |
| 預測與評估       | 計算 Accuracy、Precision、Recall、F1-score 與混淆矩陣 |
| 模型保存         | 使用 `joblib.dump` 儲存模型與向量化器至指定資料夾 |

## 執行範例

```bash
python train_predict.py --clean_csv sms_spam_clean.csv --model_dir model_artifacts
```



# 3️⃣ 實驗筆記本與視覺化

本階段主要使用 Jupyter Notebook 與 Streamlit 進行資料探索、特徵分析、模型表現觀察及交互式視覺化。

## 主要內容

### 3.1 資料探索與前處理檢查
- 顯示資料集基本資訊（行數、欄位、缺失值）
- 分析類別分佈（spam vs ham）
- 檢查文字清理後的替換 token 數量：
  - `<URL>`：網址
  - `<EMAIL>`：電子郵件
  - `<PHONE>`：電話號碼
  - `<NUM>`：數字

### 3.2 特徵分析
- 計算各類別訊息中最常出現的前 N 個 token
- 使用長條圖展示 token 頻率
- 區分 spam 與 ham 的常用詞模式

### 3.3 模型表現視覺化
- 混淆矩陣（Confusion Matrix）  
- ROC 曲線（Receiver Operating Characteristic Curve）
  - 計算 AUC（Area Under Curve）
- Precision-Recall 曲線
- 閾值掃描（Threshold Sweep）
  - 調整 spam 判定閾值，觀察 Precision、Recall、F1-score 變化
- Live Inference
  - 使用範例訊息或自訂文字，計算 spam 機率並即時顯示結果
  - 顯示機率柱狀圖與閾值標示

### 3.4 互動式視覺化工具
- **Jupyter Notebook**
  - 適合實驗筆記與初步分析
- **Streamlit**
  - 交互式儀表板
  - 支援：
    - 選擇資料集與欄位
    - 顯示類別分佈與 token 統計
    - 調整閾值並即時更新模型評估
    - Live Inference 測試訊息分類

## 範例程式碼片段

```python
# 計算混淆矩陣
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

# ROC 與 AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1], linestyle="--", color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
plt.show()

# Precision-Recall 曲線
prec, rec, _ = precision_recall_curve(y_test, y_prob)
PrecisionRecallDisplay(precision=prec, recall=rec).plot()
plt.show()
```


# 4️⃣ 部署（Streamlit 應用）

本階段將已訓練的模型部署為交互式 Web 應用，方便使用者即時測試訊息分類結果。

## 主要功能

### 4.1 介面設計
- **頁面標題與說明**
  - 顯示應用名稱與簡介
- **側邊欄 Inputs**
  - 選擇資料集（Dataset CSV）
  - 指定 Label 與 Text 欄位
  - 模型資料夾路徑（Models Dir）
  - 測試集比例（Test Size）
  - 隨機種子（Seed）
  - 決策閾值（Decision Threshold）

### 4.2 資料概覽
- 顯示類別分佈（spam / ham）
- 顯示文字清理後的 token 替換統計：
  - `<URL>`、`<EMAIL>`、`<PHONE>`、`<NUM>`

### 4.3 Token 頻率分析
- 顯示各類別最常出現的前 N 個 token
- 使用長條圖視覺化 token 分布

### 4.4 模型表現
- **Confusion Matrix**：顯示模型在測試集上的分類結果
- **ROC Curve**：觀察不同閾值下的 True Positive Rate 與 False Positive Rate
- **Precision-Recall Curve**：觀察 Precision 與 Recall 變化
- **Threshold Sweep**：透過滑桿調整閾值，查看 Precision、Recall、F1-score

### 4.5 Live Inference（即時預測）
- 使用者輸入訊息或點擊範例訊息按鈕
- 文字自動清理（normalize_text）
- 計算 spam 機率
- 顯示分類結果與 spam 機率柱狀圖
- 閾值標示直觀顯示預測判定依據

### 4.6 技術細節
- 使用 `streamlit` 套件快速建置 Web 應用
- 模型與向量化器使用 `joblib` 載入
- 預測流程：
  1. 文本正規化（文字清理、替換 token）
  2. 向量化（TF-IDF）
  3. 模型預測（Logistic Regression）
  4. 機率計算與閾值判定
- 支援即時互動與圖表刷新，方便使用者觀察模型表現

### 4.7 執行方式
```bash
# 啟動 Streamlit 應用
streamlit run app.py
```


# 5️⃣ 報告與 OpenSpec 流程說明

本階段主要記錄專案流程、分析報告與 OpenSpec 系統操作，確保專案的完整性與可追蹤性。

## 5.1 實驗報告

### 5.1.1 內容
- **資料前處理紀錄**  
  - 資料清理、缺值處理、特殊 token 替換
- **模型訓練紀錄**  
  - TF-IDF 參數設定、Logistic Regression 超參數  
  - 訓練集、測試集拆分比例  
- **模型效能指標**  
  - Accuracy、Precision、Recall、F1-score  
  - Confusion Matrix、ROC Curve、Precision-Recall Curve  
- **實驗觀察與結論**  
  - 模型對 spam/ham 分類的優劣點  
  - 閾值對 Precision/Recall/F1 的影響

### 5.1.2 可視化報告
- 使用 Jupyter Notebook 或 Streamlit 生成圖表
- 包含：
  - 類別分布長條圖
  - Top-N token 長條圖
  - Confusion Matrix
  - ROC 與 Precision-Recall 曲線
  - Threshold Sweep 表格與圖表

## 5.2 OpenSpec 流程說明

### 5.2.1 專案上傳與版本控制
- 將資料集、程式碼、模型檔案與 Notebook 上傳至 OpenSpec
- 紀錄每個版本的更新內容

### 5.2.2 任務分解
- 將專案分為多個 Task：
  1. 前處理（Preprocessing）
  2. 訓練與推論（Train & Predict）
  3. 可視化分析（Notebook/Streamlit）
  4. 部署（Streamlit 應用）
  5. 報告撰寫與 OpenSpec 紀錄

### 5.2.3 任務提交
- 每個 Task 提交時附上：
  - 程式碼與 Notebook
  - 訓練結果與模型檔案
  - 實驗筆記與圖表
  - 任務完成說明與觀察

### 5.2.4 審核與回饋
- OpenSpec 會自動生成任務報告
- 指導者或組員可在平台上給予回饋
- 依回饋調整實驗或程式碼，形成迭代優化流程

### 5.2.5 最終專案整合
- 匯整所有 Task 的 Notebook、程式碼、模型與報告
- 生成最終專案資料夾，方便展示或部署

