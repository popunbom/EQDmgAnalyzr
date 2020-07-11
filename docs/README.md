# ドキュメントについて

ドキュメントを書くことは多くのメリットがあります
- 数日後の自分が理解できるように
- 他人が理解できるように
- 自分の理解が間違ってないか整理できるように

後になってありがたみが分かるものです。多少手間だと思っても
後でつまづいたときの解決にかかる時間と比べると大したものではないと思います。

*ドキュメント、書きましょう！*


## Docstring の書き方
Python の関数、クラスなどのドキュメントは `docstring` と呼ばれます。
docstring には様々な書き方がありますが、このプロジェクトでは **numpy スタイル** で統一してます。

- [numpy スタイルでの書き方 (Qiita)](https://qiita.com/simonritchie/items/49e0813508cad4876b5a#numpy%E3%82%B9%E3%82%BF%E3%82%A4%E3%83%AB%E3%81%A7%E3%81%AE%E6%9B%B8%E3%81%8D%E6%96%B9)


## ドキュメントの自動生成

このプロジェクトでは、Sphinx というドキュメント生成ライブラリを使用しています。
コード上にドキュメントを書くことで、そこからHTMLを生成してくれます。

### 準備
- [README.md](../README.md) の **Setup** を実行しておくこと
    - `sphinx-apidoc` などのコマンドが実行できない

### make による生成
`make` で生成できるようにしています

- `make show-docs`：既存ドキュメントの削除、生成、ブラウザで開く
    - 「ブラウザで開く」は環境によっては動かない可能性あり
- `make docs`：既存ドキュメントの削除、生成
- `make clean`：既存ドキュメントの削除


### 手動による生成
- make が動かせる環境がない人向け
```bash
# リポジトリのルートディレクトリに移動する
cd EQDmgAnalyzer

# *.py ファイルから docstring を収集する  
sphinx-apidoc -f -o  ./docs/source . --separate #

# ドキュメント HTML の生成
sphinx-build -a ./docs/source ./docs/build         
```

### ドキュメントを見る
ドキュメントは `docs/build/index.html` を開くことで閲覧できます。


## Author

Fumiya Endou <endo.fumiya.14@shizuoka.ac.jp>
