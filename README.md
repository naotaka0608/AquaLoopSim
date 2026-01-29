# Fluid Simulation (Ryutai)

リアルタイム流体シミュレーションアプリケーション。
粒子法ベースの流体ソルバーをPythonで実装し、インタラクティブに操作可能です。
CPU (Numba) および GPU (CuPy) の両方での高速計算をサポートしています。

Pythonスクリプトとしての実行だけでなく、Windows用実行ファイル(.exe)へのビルドもサポートしています。

## 技術スタック (Configuration)

このプロジェクトは以下の技術で構成されています。

- **言語**: Python 3.10+
- **数値計算**:
    - **NumPy**: 基本的な配列操作
    - **Numba**: JITコンパイルによる高速計算（CPUモード）
    - **CuPy**: NVIDIA CUDAによる超高速並列計算（GPUモード）
- **可視化**:
    - **PyVista**: VTKバックエンドによる高速・高品質な3D描画
- **GUI**:
    - **Dear PyGui**: パラメータ調整用ユーザーインターフェース
- **パッケージ管理**:
    - **uv**: 高速なPythonパッケージマネージャ
- **ビルド**:
    - **PyInstaller**: スタンドアロン実行ファイルの作成

## ドキュメント (Documentation)

詳細なドキュメントは `doc/` フォルダに含まれています。

- **[ユーザーガイド (機能説明)](doc/user_guide.md)**: GPUモードの使い方やUI操作について
- **[技術仕様書 (Technical Explanation)](doc/technical_explanation.md)**: 物理モデルやアルゴリズムの詳細

## 起動方法 (Usage)

### 1. 開発環境での実行

`uv` を使用して依存関係を解決し、アプリケーションを起動します。

GPUモードを使用する場合は、CUDAバージョンに合わせた `cupy` を追加してください。

```bash
# 基本依存関係のインストール
uv sync

# (オプション) GPUモード用 CuPy のインストール
# CUDA 13.x系:
uv add cupy-cuda13x
# CUDA 12.x系:
uv add cupy-cuda12x
# (CUDA 11.x系なら cupy-cuda11x)

# アプリケーションの実行
uv run python -m src.main
```

### 2. 実行ファイル(EXE)の作成

PyInstallerを使用して、依存ライブラリを含む独立した実行ファイルを作成します。

```bash
uv run pyinstaller --noconfirm --onedir --console --name "FluidSim" --add-data "src/config.py;src" --collect-all pyvista --collect-all dearpygui src/main.py
```

- **生成場所**: `dist/FluidSim/FluidSim.exe`
- **備考**: 生成された `FluidSim` フォルダ全体を配布することで、Python環境がないWindows PCでも動作します。
