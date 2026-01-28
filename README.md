# Fluid Simulation (Ryutai)

リアルタイム流体シミュレーションアプリケーション。
粒子法ベースの流体ソルバーをPythonで実装し、インタラクティブに操作可能です。
Pythonスクリプトとしての実行だけでなく、Windows用実行ファイル(.exe)へのビルドもサポートしています。

## 技術スタック (Configuration)

このプロジェクトは以下の技術で構成されています（TaichiからNumba+PyVistaへ移行済み）。

- **言語**: Python 3.10+
- **数値計算**:
    - **NumPy**: 基本的な配列操作
    - **Numba**: JITコンパイルによる高速計算（`@njit`）
- **可視化**:
    - **PyVista**: VTKバックエンドによる高速・高品質な3D描画
- **GUI**:
    - **Dear PyGui**: パラメータ調整用ユーザーインターフェース
- **パッケージ管理**:
    - **uv**: 高速なPythonパッケージマネージャ
- **ビルド**:
    - **PyInstaller**: スタンドアロン実行ファイルの作成

## 起動方法 (Usage)

### 1. 開発環境での実行

`uv` を使用して依存関係を解決し、アプリケーションを起動します。

```bash
# 依存関係のインストール
uv sync

# アプリケーションの実行
uv run python main.py
```

### 2. 実行ファイル(EXE)の作成

PyInstallerを使用して、依存ライブラリを含む独立した実行ファイルを作成します。
NumbaやPyVista、Dear PyGuiなどの依存関係を含めるために、以下のコマンドを使用します。

```bash
uv run pyinstaller --noconfirm --onedir --console --name "FluidSim" --add-data "src/config.py;src" --collect-all pyvista --collect-all dearpygui main.py
```

- **生成場所**: `dist/FluidSim/FluidSim.exe`
- **備考**: 生成された `FluidSim` フォルダ全体を配布することで、Python環境がないWindows PCでも動作します。
