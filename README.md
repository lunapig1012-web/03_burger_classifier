<p align="center">
  <h1 align="center">🍔 Burger Classifier</h1>
  <p align="center">
    <strong>実画像・Data Augmentation・Dynamic Synthetic Data を比較する画像二値分類プロジェクト</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Task-Binary%20Classification-4C8BF5?style=flat-square" alt="Binary Classification">
    <img src="https://img.shields.io/badge/Models-ResNet18%20%7C%20MobileNetV3-34A853?style=flat-square" alt="Models">
    <img src="https://img.shields.io/badge/Device-CUDA%20%7C%20CPU-7B61FF?style=flat-square" alt="Device">
  </p>
</p>

---

## 🧠 画像二値分類と本プロジェクトの検証軸

画像二値分類は、入力画像を2つのクラスのいずれかに分類するタスクです。本プロジェクトでは、画像に Burger が含まれるかを次の固定された class order で判定します。

```text
0 = non_burger
1 = burger
```

分類モデルそのものだけでなく、**Training Data の作り方が分類性能に与える影響**を比較できるよう、3つの実験条件を用意しています。

| 実験 | Training Data | Data Augmentation | Validation / Test |
|:---:|:---|:---:|:---:|
| **Experiment A** | Real images only | `none` | 実画像のみ |
| **Experiment B** | Real images only | `basic` | 実画像のみ |
| **Experiment C** | Real images + 動的生成した Synthetic Burger images | `basic` | 実画像のみ |

> 💡 3つの実験で同じ `data/base_real/val` と `data/base_real/test` を使用し、Training Data の条件だけを変えて比較します。

---

## 📖 プロジェクト概要

**Burger Classifier** は、PyTorch による学習、検証、Test 評価、可視化までを一つの流れで実行する画像分類プロジェクトです。

主要な学習入口である `scripts/train.py` は、実画像のみの Baseline、Data Augmentation、Epoch ごとに再生成する Synthetic Data の3条件に対応しています。学習後は Validation Accuracy が最も高い Checkpoint を読み込み、共通の実画像 Test データから各種評価指標と可視化を生成します。

### ✨ 主な機能

- 🧠 **複数モデルに対応** — ResNet18、MobileNetV3 Small、MobileNetV3 Large を選択可能
- 🧪 **A / B / C 実験** — Training Data と Data Augmentation の条件を比較
- 🖼️ **実画像による共通評価** — 全実験で同じ Validation / Test ディレクトリを使用
- ✂️ **Cutout 前処理** — Labelme の Polygon Annotation から透過 Burger Cutout を生成
- 🔄 **Dynamic Synthetic Data** — Burger Cutout と背景画像から各 Epoch に学習画像を再生成
- 🎲 **再現性への配慮** — Python、NumPy、PyTorch、CUDA、DataLoader に Random Seed を設定
- 💾 **自己記述的な Checkpoint** — モデル名、class order、入力サイズ、Seed などを保存
- 📊 **複数の評価指標** — Accuracy、Precision、Recall、F1、Confusion Matrix を出力
- 🔍 **予測結果の可視化** — 正解例と誤分類例を画像グリッドとして保存
- ⚡ **Device 選択** — `--device auto` で CUDA を優先し、利用できない場合は CPU を使用

> **Note**: 現在のリポジトリには、画像データ、学習済みモデル、実行済みの評価結果は含まれていません。

---

## 🧪 実験設計

### Experiment A — Real Images Only

実画像のみを使用し、ランダムな Data Augmentation を適用しない Baseline です。Resize、Tensor 変換、ImageNet の mean / std による正規化だけを行います。

### Experiment B — Real Images + Data Augmentation

実画像に `basic` Data Augmentation を適用します。

- `RandomResizedCrop`：scale `0.75–1.0`
- `RandomHorizontalFlip`
- `RandomRotation(12)`
- `ColorJitter`
- ImageNet の mean / std による正規化

### Experiment C — Real Images + Dynamic Synthetic Data

実画像と、各 Epoch に生成する Synthetic Burger images を結合して学習します。Training Data には Experiment B と同じ `basic` Data Augmentation を適用します。

Validation / Test には、すべての実験でランダム変換を適用しません。Resize、Tensor 変換、正規化のみを使用します。

---

## 🧬 Synthetic Data Pipeline

```text
🏷️ Labelme Annotation
          ↓
✂️ Burger Cutout
          ↓
🖼️ Background Image
          ↓
🎛️ Random Transform
          ↓
🧬 Dynamic Synthetic Image
          ↓
📚 Real Training Data と結合
          ↓
🧠 Training
```

### 1. Burger Cutout の作成

Labelme で `burger` の Polygon Annotation を作成します。`labelme_to_cutout.py` は Polygon から Alpha Mask を生成し、境界を Feathering した透過 PNG を保存します。

### 2. Synthetic Image の生成

各 Epoch の開始時に `data/synthetic/current_epoch` を作り直し、次の処理を行います。

1. Burger Cutout と背景画像をランダムに選択
2. Cutout の Scale と Rotation をランダムに変更
3. Alpha 境界を Feathering
4. Brightness と Contrast をランダムに変更
5. 背景上のランダムな位置に配置
6. 生成画像を実画像 Training Data と結合

既定の生成枚数は1 Epoch あたり50枚です。生成時には `seed + epoch` を使用し、Epoch ごとに異なる画像を生成しながら、同じ Seed を指定した処理の再現性に配慮しています。

---

## 🛠️ 技術スタック

| カテゴリ | 技術 |
|:---|:---|
| **言語** | Python |
| **Deep Learning** | PyTorch、TorchVision |
| **Model** | ResNet18、MobileNetV3 Small、MobileNetV3 Large |
| **Image Processing** | Pillow、NumPy |
| **Visualization** | Matplotlib |
| **Progress Display** | tqdm |
| **Annotation** | Labelme |

Labelme は Annotation の作成に使用しますが、`requirements.txt` には含まれていません。

---

## 📂 プロジェクト構成

```text
burger_classifier-master/
├── 📁 scripts/
│   ├── train.py                  # A / B / C の主要学習入口
│   ├── evaluate.py               # 保存済みモデルの Test 評価
│   ├── visualize_predictions.py  # 正解例・誤分類例の可視化
│   ├── labelme_to_cutout.py      # Labelme JSON から透過 Cutout を生成
│   ├── generate_fake_data.py     # Synthetic Data の生成処理
│   ├── generate_synthetic.py     # Synthetic Data の生成入口
│   ├── train_dynamic_fake.py     # Dynamic Synthetic Data 用の別学習入口
│   ├── rename_images.py          # 画像ファイル名の変更
│   └── utils.py                  # Model、Device、Seed、ファイル操作
│
├── 📄 requirements.txt
└── 📄 README.md
```

次のディレクトリは、データ準備または実行時に作成します。現在のリポジトリには含まれていません。

```text
data/
cutouts/
backgrounds/
outputs/
```

---

## 🚀 クイックスタート

### 前提条件

- Python と pip が利用できること
- 学習用、Validation 用、Test 用の実画像が準備されていること
- Experiment C を実行する場合は、Burger Cutout と背景画像が準備されていること
- CUDA 対応 GPU は任意。利用できない場合は CPU で実行可能

### インストール

本 README があるプロジェクトルートで実行します。

```powershell
# 1. 仮想環境を作成
python -m venv .venv

# 2. 仮想環境を有効化
.\.venv\Scripts\Activate.ps1

# 3. 依存 Package をインストール
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

事前学習済みモデルを初めて使用する場合、TorchVision が対応する Weight をダウンロードします。

### データの配置

```text
data/
└── base_real/
    ├── train/
    │   ├── burger/
    │   └── non_burger/
    ├── val/
    │   ├── burger/
    │   └── non_burger/
    └── test/
        ├── burger/
        └── non_burger/
```

対応形式は `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp` です。6つの class ディレクトリには、それぞれ1枚以上の画像が必要です。

### Burger Cutout の作成

```powershell
python scripts/labelme_to_cutout.py `
  --json_dir labelme_json `
  --image_dir data/base_real/train/burger `
  --output_dir cutouts/burger `
  --label burger `
  --feather_radius 2.0
```

### Experiment A を実行

```powershell
python scripts/train.py `
  --data_dir data/base_real `
  --exp_name exp_A_real_only `
  --augmentation none `
  --model resnet18
```

### Experiment B を実行

```powershell
python scripts/train.py `
  --data_dir data/base_real `
  --exp_name exp_B_real_aug `
  --augmentation basic `
  --model resnet18
```

### Experiment C を実行

Experiment C では、次の追加ディレクトリを使用します。

```text
cutouts/
└── burger/     # 透過 Burger Cutout

backgrounds/    # 背景画像
```

```powershell
python scripts/train.py `
  --data_dir data/base_real `
  --dynamic_synthetic `
  --cutout_dir cutouts/burger `
  --background_dir backgrounds `
  --fake_per_epoch 50 `
  --exp_name exp_C_real_synthetic `
  --augmentation basic `
  --model resnet18
```

---

## ⚙️ 設定

### 主な学習オプション

| オプション | 既定値 | 説明 |
|:---|:---:|:---|
| `--model` | `resnet18` | `resnet18`、`mobilenet_v3_small`、`mobilenet_v3_large` |
| `--epochs` | `10` | 学習 Epoch 数 |
| `--batch_size` | `32` | Batch Size |
| `--lr` | `0.0001` | Adam Optimizer の Learning Rate |
| `--seed` | `42` | Random Seed |
| `--image_size` | `224` | 入力画像サイズ |
| `--num_workers` | `0` | DataLoader の Worker 数 |
| `--device` | `auto` | CUDA / CPU の自動選択、または PyTorch Device の明示指定 |
| `--no_pretrained` | 無効 | 指定時は ImageNet の事前学習済み Weight を使用しない |

### Checkpoint の内容

`best_model.pth` には次の情報を保存します。

```text
model_state_dict
model_name
classes
class_to_idx
image_size
best_epoch
best_val_accuracy
augmentation
seed
```

`evaluate.py` と `visualize_predictions.py` は、Checkpoint の `classes` が `['non_burger', 'burger']` と一致することを確認してからモデルを読み込みます。

---

## 🏗️ アーキテクチャ

```text
┌──────────────────────────────────────────────────────────────┐
│                     Training Pipeline                        │
│                                                              │
│  📁 Real Train Images                                        │
│          │                                                   │
│          ├──────────── Experiment A: augmentation = none     │
│          │                                                   │
│          ├──────────── Experiment B: augmentation = basic    │
│          │                                                   │
│          └── Experiment C                                    │
│                ↑                                             │
│      Burger Cutout + Background                              │
│                ↓                                             │
│      Dynamic Synthetic Images                                │
│                                                              │
│  Training Data                                               │
│          ↓                                                   │
│  ResNet18 / MobileNetV3                                      │
│          ↓                                                   │
│  Real Validation Data ──→ Best Checkpoint                    │
│                                  ↓                           │
│                           Real Test Data                      │
│                                  ↓                           │
│        Metrics / Confusion Matrix / Prediction Examples      │
└──────────────────────────────────────────────────────────────┘
```

Validation Accuracy が更新された時点で `best_model.pth` を保存します。全 Epoch の終了後、その Checkpoint を再ロードして Test データを評価します。

---

## 📊 評価

`burger` を Positive class として、次の指標を算出します。

| 指標 | 内容 |
|:---|:---|
| **Accuracy** | 全 Test サンプルのうち正しく分類した割合 |
| **Precision** | `burger` と予測したうち、実際に `burger` だった割合 |
| **Recall** | 実際の `burger` のうち、正しく `burger` と判定した割合 |
| **F1** | Precision と Recall の調和平均 |
| **Confusion Matrix** | True label と Predicted label の組み合わせ別件数 |

保存済み Checkpoint は個別に再評価できます。

```powershell
python scripts/evaluate.py `
  --data_dir data/base_real `
  --model_path outputs/exp_C_real_synthetic/best_model.pth `
  --output_dir outputs/exp_C_real_synthetic/evaluation
```

現在のリポジトリには実行済みの `metrics.json` がないため、実測値は掲載していません。

| 実験 | Accuracy | Precision | Recall | F1 | Best Epoch |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Experiment A** | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 |
| **Experiment B** | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 |
| **Experiment C** | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 |

---

## 📦 出力ファイル

`scripts/train.py` は実験結果を `outputs/{exp_name}/` に保存します。

```text
outputs/{exp_name}/
├── best_model.pth
├── metrics.json
├── history.json
├── train_loss.png
├── val_accuracy.png
├── confusion_matrix.png
├── correct_examples.png
└── wrong_examples.png
```

| ファイル | 内容 |
|:---|:---|
| `best_model.pth` | Validation Accuracy が最も高かった Checkpoint |
| `metrics.json` | Test 指標、Confusion Matrix、実験設定、データ件数 |
| `history.json` | Epoch ごとの Training / Validation 記録 |
| `train_loss.png` | Epoch ごとの **Training Loss のみ**を示す曲線 |
| `val_accuracy.png` | Epoch ごとの Validation Accuracy 曲線 |
| `confusion_matrix.png` | Test データの Confusion Matrix |
| `correct_examples.png` | 正しく分類した Test サンプルの一覧 |
| `wrong_examples.png` | 誤分類した Test サンプルの一覧 |

`correct_examples.png` と `wrong_examples.png` は、それぞれ該当するサンプルが1件以上存在する場合のみ生成されます。`evaluate.py` を単独実行した場合は、`test_metrics.json` と `confusion_matrix.png` を保存します。

---

## ⚠️ データ分割時の注意

同一シーンの連続撮影画像や、同一原画像から作成した派生画像を Train / Validation / Test に跨がせると、データリークにつながる可能性があります。

この分割はコードによって自動保証されません。データ準備時に、撮影単位または原画像単位で分割してください。また、Synthetic Data は Training Data のみに使用し、Validation / Test には実画像だけを配置してください。

---

## 🚧 制約と今後の改善

### 現在の制約

- 画像データ、学習済み Weight、実行済み評価結果はリポジトリに含まれていない
- Synthetic Data は前景を背景へ合成する方式であり、照明、影、奥行き、遮蔽の表現に制約がある
- 画像単位の二値分類であり、画像内の Burger の位置は検出しない
- 撮影元や原画像単位の重複を自動検査しない
- クラス不均衡を自動補正する Sampler や Class Weight は実装していない
- `requirements.txt` の Package Version は固定していない

### 今後の改善候補

- 同一条件で実行した A / B / C の定量比較
- 外部 Test Set を用いた汎化性能の検証
- Grad-CAM などによる判断根拠の可視化
- Synthetic Data の影、透視変換、遮蔽表現の改善
- Decision Threshold の調整とクラス不均衡への対応
- ONNX などの推論形式への Export
- Automated Test と CI による学習・評価処理の検証

---

## Origin / Attribution

現在のコード一式には、確認可能な上流リポジトリの URL、原作者情報、または Commit History が含まれていません。そのため、本 README では出典を推測して記載していません。

第三者のコードを基にした変更が含まれる場合は、公開前に原プロジェクト、原作者、対象箇所、および本プロジェクトで変更・追加した範囲を確認し、正確な Attribution を追記してください。

## License Note

現在のコード一式には `LICENSE` ファイルが含まれていません。ライセンスが明示されていないことは、自由な複製、変更、再配布を許可するものではありません。

公開、再配布、または商用利用を行う場合は、ソースコード、画像データ、背景素材、Annotation、事前学習済み Weight について、それぞれの利用条件を確認し、適切なライセンス情報を追加してください。
