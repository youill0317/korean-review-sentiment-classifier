# Best_CV_BinaryCount_Stacking_MCC07667.ipynb 코드 상세 해설

원본 노트북: `mid/Best_CV_BinaryCount_Stacking_MCC07667.ipynb`

이 문서는 원본 노트북의 코드 셀을 순서대로 옮기고, 각 코드 셀의 목적, 설계 이유, 내부 동작, 알고리즘적 의미, 주의점을 자세히 설명한 파일입니다.
코드 블록 안의 내용은 원본 `.ipynb` 코드 셀 내용을 그대로 가져온 것입니다.

## 전체 요약

이 노트북은 한국어 감성분류를 위해 sklearn 기반의 고차원 텍스트 feature와 선형 모델 스태킹을 사용합니다. 핵심 아이디어는 형태소 분석이나 딥러닝 없이 문자 n-gram과 단어 n-gram을 넓게 추출한 뒤, 여러 선형 분류기의 판단을 `StackingClassifier`로 결합하는 것입니다. 평가는 MCC를 기준으로 하며, 최종 산출물은 `final_pipeline.pkl`과 `submission.csv`입니다.

## Code Cell 1 (Notebook Cell 1)

```python
from pathlib import Path
import os
import re
import time
import html
import unicodedata

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
print("Environment ready")
```

### 상세 해설

이 셀은 전체 노트북의 실행 기반을 준비하는 초기화 셀입니다. 파일 경로 처리, 정규표현식 전처리, 시간 측정, HTML escape 해제, 유니코드 정규화, 데이터 처리, 모델 저장, 머신러닝 모델 구성에 필요한 도구를 모두 불러옵니다.

`pathlib.Path`는 Colab과 Windows 로컬처럼 경로 표기가 다른 환경에서도 안정적으로 파일 경로를 다루기 위해 사용합니다. `os`는 `DATA_DIR` 같은 환경변수를 읽기 위해 필요합니다. `re`, `html`, `unicodedata`는 텍스트 전처리에 쓰입니다. 특히 한국어 리뷰 데이터에는 HTML 태그, URL, 줄바꿈, 전각 문자, 특수문자 표현이 섞일 수 있으므로 정규화 과정이 중요합니다.

`joblib`은 학습된 sklearn 파이프라인을 `pkl` 파일로 저장하는 데 사용됩니다. 이 노트북은 전처리기, 벡터라이저, TF-IDF 변환기, 분류기를 하나의 `Pipeline`으로 묶기 때문에, `joblib.dump`로 저장하면 나중에 원문 텍스트를 바로 넣어 예측할 수 있는 완성 모델이 됩니다.

`sklearn`에서 가져온 클래스들은 이 노트북의 모델링 전략을 보여 줍니다. `CountVectorizer`로 문자/단어 n-gram을 만들고, `TfidfTransformer`로 구분력 있는 패턴에 가중치를 주며, `LinearSVC`, `RidgeClassifier`, `LogisticRegression`, `ComplementNB` 같은 선형 모델을 여러 개 학습합니다. 마지막에는 `StackingClassifier`가 이 모델들의 판단을 종합합니다.

`RANDOM_STATE = 42`는 재현성을 위한 고정 난수입니다. 데이터를 섞거나 모델 내부에서 난수가 쓰이는 경우 같은 결과가 나오도록 도와줍니다. 교육용 과제와 리더보드 제출에서는 같은 노트북을 다시 실행했을 때 결과가 크게 흔들리지 않는 것이 중요합니다.

핵심 특징은 딥러닝이나 형태소 분석기가 아니라 sklearn 기반의 고차원 희소 텍스트 벡터와 선형 분류기를 사용한다는 점입니다. 한국어 감성분류에서는 `좋`, `별로`, `최악`, `ㅋㅋ`, `ㅠㅠ`, `안좋` 같은 짧은 문자 패턴이 강력한 신호가 될 수 있으므로 문자 n-gram 방식이 매우 실용적입니다.

## Code Cell 2 (Notebook Cell 2)

```python
# Colab default: put public_train.csv and public_test.csv in /content/drive/MyDrive/UD_26
# Local default: this repository keeps the files in ./mid

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB and not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")

DATA_DIR_CANDIDATES = []
if os.environ.get("DATA_DIR"):
    DATA_DIR_CANDIDATES.append(Path(os.environ["DATA_DIR"]))

cwd = Path.cwd()
DATA_DIR_CANDIDATES.extend([
    Path("/content/drive/MyDrive/UD_26"),
    Path("/content/drive/MyDrive/UD_26/mid"),
    cwd,
    cwd / "mid",
    cwd / "data",
    cwd.parent,
    cwd.parent / "mid",
])

# Remove duplicates while preserving order.
seen = set()
DATA_DIR_CANDIDATES = [p for p in DATA_DIR_CANDIDATES if not (str(p) in seen or seen.add(str(p)))]

DATA_DIR = None
for candidate in DATA_DIR_CANDIDATES:
    if (candidate / "public_train.csv").exists() and (candidate / "public_test.csv").exists():
        DATA_DIR = candidate
        break

if DATA_DIR is None:
    checked = "\n".join(str(p) for p in DATA_DIR_CANDIDATES)
    raise FileNotFoundError(
        "Could not find public_train.csv and public_test.csv.\n"
        "Colab: run this after Drive is mounted and place the files in /content/drive/MyDrive/UD_26 or /content/drive/MyDrive/UD_26/mid.\n"
        "Local: run from the UD_26 repository root or from the mid folder, or set DATA_DIR to the folder containing the CSV files.\n"
        f"Current working directory: {cwd}\n"
        f"Checked paths:\n{checked}"
    )

TRAIN_PATH = DATA_DIR / "public_train.csv"
TEST_PATH = DATA_DIR / "public_test.csv"
OUTPUT_DIR = DATA_DIR
MODEL_DIR = OUTPUT_DIR / "model" / "Best_CV_BinaryCount_Stacking"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = MODEL_DIR / "final_pipeline.pkl"
SUBMISSION_PATH = MODEL_DIR / "submission.csv"

print("IN_COLAB:", IN_COLAB)
print("Current working directory:", cwd)
print("DATA_DIR:", DATA_DIR.resolve())
print("TRAIN_PATH:", TRAIN_PATH.resolve())
print("TEST_PATH:", TEST_PATH.resolve())
print("MODEL_DIR:", MODEL_DIR.resolve())
```

### 상세 해설

이 셀은 학습 데이터와 테스트 데이터가 있는 폴더를 자동으로 찾고, 모델과 제출 파일을 저장할 경로를 설정합니다. Colab과 로컬 실행을 모두 고려한 환경 설정 코드입니다.

먼저 `google.colab` 모듈을 import할 수 있는지 확인해서 현재 환경이 Colab인지 판단합니다. Colab이면 Google Drive가 마운트되어 있는지 확인하고, 아직 마운트되지 않았다면 `drive.mount("/content/drive")`를 실행합니다. 과제 지침상 Colab에서는 `/content/drive/MyDrive/UD_26` 경로가 기본 데이터 폴더로 사용되기 때문에 이 처리가 필요합니다.

`DATA_DIR_CANDIDATES`는 데이터가 있을 만한 후보 경로 목록입니다. 환경변수 `DATA_DIR`가 있으면 가장 먼저 확인합니다. 그 다음 Colab Drive 경로, Drive 안의 `mid` 폴더, 현재 작업 폴더, 현재 폴더의 `mid`, `data`, 부모 폴더 등을 차례로 확인합니다. 이렇게 하면 사용자가 노트북을 저장소 루트에서 실행하든, `mid` 폴더 안에서 실행하든, Colab Drive에서 실행하든 파일을 찾을 가능성이 높아집니다.

중복 제거 코드는 후보 경로가 반복되는 것을 막습니다. 같은 경로를 여러 번 검사해도 큰 문제는 없지만, 에러 메시지와 출력이 지저분해지므로 순서를 유지하면서 한 번만 남깁니다.

핵심 탐색 알고리즘은 각 후보 폴더에 `public_train.csv`와 `public_test.csv`가 둘 다 있는지 검사하는 것입니다. 둘 중 하나만 있으면 올바른 과제 데이터 폴더가 아니므로 선택하지 않습니다. 아무 후보에서도 찾지 못하면, 어떤 경로를 확인했는지 모두 보여 주는 `FileNotFoundError`를 발생시킵니다. 이 에러 메시지는 사용자가 데이터 위치 문제를 직접 해결할 수 있도록 설계되어 있습니다.

마지막으로 `TRAIN_PATH`, `TEST_PATH`, `OUTPUT_DIR`, `MODEL_DIR`, `FINAL_MODEL_PATH`, `SUBMISSION_PATH`를 정의합니다. `MODEL_DIR.mkdir(parents=True, exist_ok=True)`는 저장 폴더가 없으면 만들어 줍니다. 이 셀의 목적은 이후 셀들이 파일 경로를 신경 쓰지 않고 학습과 저장에 집중할 수 있게 만드는 것입니다.

## Code Cell 3 (Notebook Cell 3)

```python
train = pd.read_csv(TRAIN_PATH, encoding="utf-8")
test = pd.read_csv(TEST_PATH, encoding="utf-8")

print("train shape:", train.shape)
print("test shape:", test.shape)
print("label counts:")
print(train["label"].value_counts())

assert list(train.columns) == ["row_id", "text", "label"]
assert list(test.columns) == ["row_id", "text"]
assert set(train["label"].unique()) == {"POSITIVE", "NEGATIVE"}
assert train["text"].isna().sum() == 0
assert test["text"].isna().sum() == 0

X = train["text"].astype(str)
y = train["label"].astype(str)
X_test = test["text"].astype(str)
```

### 상세 해설

이 셀은 CSV 파일을 읽고, 데이터가 과제에서 기대하는 형식인지 검증한 뒤, 모델 입력 변수를 준비합니다.

`pd.read_csv(..., encoding="utf-8")`는 한국어 텍스트가 포함된 CSV를 UTF-8로 읽습니다. 이후 `train.shape`, `test.shape`, `train["label"].value_counts()`를 출력해 데이터 크기와 라벨 분포를 확인합니다. 라벨 분포는 감성분류에서 중요합니다. 한쪽 라벨이 너무 많으면 단순 accuracy는 높아도 실제 분류 품질이 낮을 수 있고, 그래서 이 노트북은 MCC를 평가 지표로 사용합니다.

여러 `assert`는 데이터 스키마를 보호하는 안전장치입니다. 학습 데이터 열은 정확히 `row_id`, `text`, `label`이어야 하고, 테스트 데이터 열은 `row_id`, `text`이어야 합니다. 열 이름이나 순서가 다르면 제출 파일 생성에서 row_id가 틀어지거나 라벨 열을 잘못 읽을 수 있습니다.

`assert set(train["label"].unique()) == {"POSITIVE", "NEGATIVE"}`는 라벨이 정확히 두 문자열인지 확인합니다. 과제 규칙상 라벨을 숫자 0/1로 임의 변환하면 제출 과정에서 역매핑 실수가 생길 수 있으므로 문자열 라벨을 그대로 유지합니다.

텍스트 결측치 검사도 중요합니다. `CountVectorizer`는 기본적으로 문자열 입력을 기대합니다. 결측치가 있으면 학습 중 오류가 나거나 예상치 못한 전처리가 발생할 수 있습니다.

마지막의 `X`, `y`, `X_test`는 모델링에 쓰는 핵심 변수입니다. `X`는 학습 텍스트, `y`는 정답 라벨, `X_test`는 제출 예측을 만들 테스트 텍스트입니다. 모두 `astype(str)`로 문자열화하여 벡터라이저가 안정적으로 처리할 수 있게 합니다. 이 셀은 모델을 학습하기 전 데이터 계약을 확정하는 단계입니다.

## Code Cell 4 (Notebook Cell 4)

```python
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, mode="raw", lowercase=True):
        self.mode = mode
        self.lowercase = lowercase

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._one(x) for x in X]

    def _base(self, x):
        x = "" if x is None else str(x)
        x = unicodedata.normalize("NFKC", html.unescape(x))
        x = re.sub(r"<[^>]+>", " ", x)
        x = re.sub(r"https?://\S+|www\.\S+", " URL ", x)
        x = re.sub(r"\s+", " ", x).strip()
        if self.lowercase:
            x = x.lower()
        return x

    def _one(self, x):
        x = self._base(x)
        if self.mode == "raw":
            return x
        if self.mode == "nospace":
            return re.sub(r"\s+", "", x)
        raise ValueError(f"unknown mode: {self.mode}")
```

### 상세 해설

이 셀은 sklearn 파이프라인 안에서 사용할 사용자 정의 텍스트 전처리기 `TextPreprocessor`를 정의합니다. 일반 함수가 아니라 `BaseEstimator`, `TransformerMixin`을 상속한 클래스로 만든 이유는 전처리 과정을 모델 파이프라인 안에 포함하고, 나중에 `joblib`으로 함께 저장하기 위해서입니다.

`__init__`의 `mode`는 전처리 결과를 어떤 형태로 반환할지 결정합니다. `raw`는 공백을 유지한 기본 텍스트이고, `nospace`는 모든 공백을 제거한 텍스트입니다. 한국어 리뷰에서는 띄어쓰기가 정확하지 않은 경우가 많기 때문에, 공백 제거 버전은 `진짜 좋아요`, `진짜좋아요`, `진짜  좋아요` 같은 표현을 비슷한 문자 n-gram으로 처리하는 데 도움이 됩니다.

`fit`은 아무것도 학습하지 않고 `self`를 반환합니다. 이 전처리기는 평균이나 vocabulary처럼 데이터에서 배워야 할 값이 없기 때문입니다. 그래도 sklearn 파이프라인 규약상 `fit` 메서드가 필요합니다.

`transform`은 입력 목록의 각 텍스트에 `_one`을 적용해 전처리된 문자열 리스트를 만듭니다. 이 결과가 바로 `CountVectorizer`의 입력이 됩니다.

`_base`는 실제 정리 작업을 수행합니다. `None`은 빈 문자열로 바꾸고, 모든 입력은 문자열로 변환합니다. `html.unescape`는 `&amp;` 같은 HTML escape를 원래 문자로 되돌립니다. `unicodedata.normalize("NFKC", ...)`는 전각/반각 문자나 호환 문자를 더 표준적인 형태로 정리합니다.

정규표현식 `re.sub(r"<[^>]+>", " ", x)`는 HTML 태그를 공백으로 제거합니다. `re.sub(r"https?://\S+|www\.\S+", " URL ", x)`는 실제 URL 문자열을 공통 토큰 `URL`로 치환합니다. URL 주소 자체는 너무 희소하지만 URL이 있다는 사실은 정보일 수 있기 때문입니다.

`re.sub(r"\s+", " ", x).strip()`는 여러 공백, 줄바꿈, 탭을 하나의 공백으로 줄이고 앞뒤 공백을 제거합니다. `lowercase=True`이면 영문자를 소문자로 통일합니다. 한국어에는 대소문자가 없지만, 영어 단어나 URL 주변 토큰에는 효과가 있습니다.

이 전처리기의 핵심은 형태소 분석이 아니라 표면 문자열을 안정화하는 것입니다. 문자 n-gram 모델은 오타, 이모티콘, 반복 문자, 어미 변화, 띄어쓰기 차이에 비교적 강하므로, 이 정도의 정규화만으로도 감성분류에 유용한 입력을 만들 수 있습니다.

## Code Cell 5 (Notebook Cell 5)

```python
BASE_CONFIG = {
    "tag": "stack_binarycount_svc0p12_0p25_0p5_ridge2_logreg3_cnb0005_meta2_rs42",
    "svc_c_low": 0.12,
    "svc_c_mid": 0.25,
    "svc_c_high": 0.50,
    "ridge_alpha": 2.0,
    "logreg_c": 3.0,
    "cnb_alpha": 0.005,
    "meta_c": 2.0,
}

SVC_LOW_VALUES = [0.08, 0.10, 0.12]
SVC_MID_VALUES = [0.23, 0.25, 0.27]
SVC_HIGH_VALUES = [0.48, 0.50, 0.52]
RIDGE_ALPHA_VALUES = [1.5, 2.0, 2.5]
LOGREG_C_VALUES = [2.0, 3.0, 4.0]
CNB_ALPHA_VALUES = [0.003, 0.005, 0.008]
META_C_VALUES = [1.5, 2.0, 2.5]

SELECTED_CONFIG = BASE_CONFIG.copy()
print("Default config:", SELECTED_CONFIG)


def config_tag(config):
    return (
        f"stack_binarycount_svc{str(config['svc_c_low']).replace('.', 'p')}_"
        f"{str(config['svc_c_mid']).replace('.', 'p')}_"
        f"{str(config['svc_c_high']).replace('.', 'p')}_"
        f"ridge{str(config['ridge_alpha']).replace('.', 'p')}_"
        f"logreg{str(config['logreg_c']).replace('.', 'p')}_"
        f"cnb{str(config['cnb_alpha']).replace('.', 'p')}_"
        f"meta{str(config['meta_c']).replace('.', 'p')}_rs42"
    )


def build_best_cv_pipeline(config=None):
    if config is None:
        config = SELECTED_CONFIG

    features = Pipeline([
        ("features", FeatureUnion([
            ("nospace_char", Pipeline([
                ("prep", TextPreprocessor(mode="nospace", lowercase=True)),
                ("vec", CountVectorizer(
                    analyzer="char",
                    ngram_range=(1, 6),
                    min_df=2,
                    max_features=800_000,
                    lowercase=False,
                    binary=True,
                    dtype=np.float32,
                )),
            ])),
            ("raw_char_wb", Pipeline([
                ("prep", TextPreprocessor(mode="raw", lowercase=True)),
                ("vec", CountVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 6),
                    min_df=2,
                    max_features=500_000,
                    lowercase=False,
                    binary=True,
                    dtype=np.float32,
                )),
            ])),
            ("raw_word", Pipeline([
                ("prep", TextPreprocessor(mode="raw", lowercase=True)),
                ("vec", CountVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=300_000,
                    token_pattern=r"(?u)\b\w+\b",
                    lowercase=False,
                    binary=True,
                    dtype=np.float32,
                )),
            ])),
        ], n_jobs=1)),
        ("tfidf", TfidfTransformer(
            sublinear_tf=False,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
        )),
    ])

    base_estimators = [
        ("svc_low", LinearSVC(C=config["svc_c_low"], random_state=RANDOM_STATE, dual="auto", max_iter=3000)),
        ("svc_mid", LinearSVC(C=config["svc_c_mid"], random_state=RANDOM_STATE, dual="auto", max_iter=3000)),
        ("svc_high", LinearSVC(C=config["svc_c_high"], random_state=RANDOM_STATE, dual="auto", max_iter=3000)),
        ("ridge", RidgeClassifier(alpha=config["ridge_alpha"])),
        ("logreg", LogisticRegression(C=config["logreg_c"], solver="liblinear", random_state=RANDOM_STATE, max_iter=1000)),
        ("cnb", ComplementNB(alpha=config["cnb_alpha"])),
    ]

    classifier = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=config["meta_c"], solver="liblinear", random_state=RANDOM_STATE, max_iter=1000),
        cv=3,
        stack_method="auto",
        passthrough=True,
        n_jobs=1,
    )

    return Pipeline([
        ("features", features),
        ("classifier", classifier),
    ])

pipeline = build_best_cv_pipeline()
pipeline
```

### 상세 해설

이 셀은 노트북의 핵심 모델을 만드는 `build_best_cv_pipeline` 함수를 정의합니다. 이 함수는 텍스트 전처리, 세 종류의 벡터화, TF-IDF 변환, 여러 기본 분류기, 최종 스태킹 분류기를 모두 하나의 sklearn `Pipeline`으로 묶습니다.

`BASE_CONFIG`는 기본 하이퍼파라미터 모음입니다. 세 개의 LinearSVC C 값, RidgeClassifier alpha, LogisticRegression C, ComplementNB alpha, 메타 LogisticRegression C가 들어 있습니다. `C`는 LogisticRegression과 LinearSVC에서 정규화의 역수입니다. C가 클수록 학습 데이터에 더 강하게 맞추고, C가 작을수록 가중치를 더 억제합니다. Ridge의 `alpha`는 반대로 클수록 정규화가 강합니다. ComplementNB의 `alpha`는 확률 추정에서 0 확률을 막는 smoothing 값입니다.

`config_tag` 함수는 하이퍼파라미터 값을 파일명에 들어갈 수 있는 설명적인 문자열로 바꿉니다. 예를 들어 `0.12`를 `0p12`로 바꿔 모델 파일명에서 점을 피합니다. 이렇게 하면 파일 이름만 봐도 어떤 설정으로 만든 모델인지 알 수 있습니다.

feature 생성부는 `FeatureUnion`을 사용합니다. `FeatureUnion`은 여러 변환기를 병렬로 적용한 뒤 결과 feature matrix를 가로로 이어 붙입니다. 여기서는 `nospace_char`, `raw_char_wb`, `raw_word` 세 관점을 동시에 사용합니다.

`nospace_char`는 공백을 제거한 텍스트에서 문자 1~6그램을 만듭니다. 한국어 감성 표현은 짧은 문자 조각에 강한 신호가 많습니다. 예를 들어 `좋`, `좋아`, `별로`, `최악`, `ㅠㅠ`, `ㅋㅋ` 같은 패턴이 직접적인 감성 단서가 됩니다. 공백 제거는 띄어쓰기 오류에 강하게 만듭니다.

`raw_char_wb`는 공백을 유지한 텍스트에서 `char_wb` 방식의 문자 2~6그램을 만듭니다. `char_wb`는 단어 경계를 고려하므로 일반 문자 n-gram보다 단어 내부와 주변 경계 정보를 더 잘 반영합니다. 공백 제거 feature와 서로 보완적입니다.

`raw_word`는 단어 1~2그램을 만듭니다. 문자 n-gram이 세밀한 표면 패턴을 잡는다면, 단어 n-gram은 `배송 빠름`, `다시 구매`, `품질 별로`처럼 더 큰 표현 단위를 잡습니다. `token_pattern=r"(?u)\b\w+\b"`는 한 글자 토큰도 허용합니다.

각 CountVectorizer의 `binary=True`는 등장 횟수 대신 등장 여부만 사용한다는 뜻입니다. 같은 표현이 한 문서에 여러 번 반복되어도 1로 처리합니다. 이후 `TfidfTransformer`가 전체 문서에서 흔한 feature의 가중치를 낮추고 구분적인 feature의 가중치를 높입니다. 즉, 이 모델은 "많이 반복되었는가"보다 "이 표현이 등장했는가"와 "전체 데이터에서 얼마나 구분적인가"를 중시합니다.

기본 분류기는 여섯 개입니다. C 값이 다른 `LinearSVC` 세 개는 같은 feature를 보더라도 조금씩 다른 결정 경계를 만듭니다. `RidgeClassifier`는 L2 정규화 기반의 다른 선형 손실을 사용합니다. `LogisticRegression`은 로지스틱 손실 기반의 선형 모델입니다. `ComplementNB`는 Naive Bayes 계열로, 텍스트 분류에서 선형 모델과 다른 관점을 제공합니다.

`StackingClassifier`는 기본 모델들의 예측을 다시 최종 모델이 학습하는 앙상블입니다. 내부적으로 기본 모델들의 out-of-fold 예측을 만들고, 그 예측을 입력으로 메타 모델을 학습합니다. `final_estimator`는 LogisticRegression입니다. `cv=3`은 스태킹 내부 예측을 만들 때 3-fold를 사용한다는 뜻입니다.

`passthrough=True`가 매우 중요합니다. 이 설정은 메타 모델이 기본 모델들의 예측값뿐 아니라 원래 TF-IDF feature도 함께 보게 합니다. 따라서 최종 LogisticRegression은 "기본 모델들이 어떻게 판단했는지"와 "원문 feature가 무엇인지"를 동시에 사용합니다. 성능 향상을 기대할 수 있지만, feature 차원이 커져 학습 비용도 늘어납니다.

최종적으로 반환되는 Pipeline은 원문 텍스트 입력부터 최종 라벨 예측까지 전 과정을 포함합니다. 따라서 저장된 `final_pipeline.pkl`은 별도 전처리 코드 없이 raw text를 바로 예측할 수 있는 완성형 모델입니다.

## Code Cell 6 (Notebook Cell 6)

```python
RUN_TUNING = True


def evaluate_config(config, cv, scorer, stage):
    config = config.copy()
    config["tag"] = config_tag(config)
    scores = cross_val_score(
        build_best_cv_pipeline(config),
        X,
        y,
        scoring=scorer,
        cv=cv,
        n_jobs=1,
    )
    row = {
        "stage": stage,
        "tag": config["tag"],
        "mean_mcc": scores.mean(),
        "std_mcc": scores.std(),
        "fold_scores": scores,
        "config": config,
    }
    print(f"[{stage}] {config['tag']}: mean={scores.mean():.6f}, std={scores.std():.6f}, scores={scores}")
    return row


if RUN_TUNING:
    start = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mcc_scorer = make_scorer(matthews_corrcoef)
    tuning_rows = []
    current_config = BASE_CONFIG.copy()

    svc_candidates = []
    for low_c in SVC_LOW_VALUES:
        for mid_c in SVC_MID_VALUES:
            for high_c in SVC_HIGH_VALUES:
                candidate = current_config.copy()
                candidate.update({"svc_c_low": low_c, "svc_c_mid": mid_c, "svc_c_high": high_c})
                svc_candidates.append(candidate)
    stage_rows = [evaluate_config(candidate, cv, mcc_scorer, "svc") for candidate in svc_candidates]
    tuning_rows.extend(stage_rows)
    current_config = max(stage_rows, key=lambda row: row["mean_mcc"])["config"].copy()
    print("Selected after svc:", current_config)

    for stage, key, values in [
        ("ridge", "ridge_alpha", RIDGE_ALPHA_VALUES),
        ("logreg", "logreg_c", LOGREG_C_VALUES),
        ("cnb", "cnb_alpha", CNB_ALPHA_VALUES),
        ("meta", "meta_c", META_C_VALUES),
    ]:
        stage_candidates = []
        for value in values:
            candidate = current_config.copy()
            candidate[key] = value
            stage_candidates.append(candidate)
        stage_rows = [evaluate_config(candidate, cv, mcc_scorer, stage) for candidate in stage_candidates]
        tuning_rows.extend(stage_rows)
        current_config = max(stage_rows, key=lambda row: row["mean_mcc"])["config"].copy()
        print(f"Selected after {stage}:", current_config)

    tuning_results = pd.DataFrame(tuning_rows).sort_values("mean_mcc", ascending=False).reset_index(drop=True)
    SELECTED_CONFIG = current_config.copy()
    SELECTED_CONFIG["tag"] = config_tag(SELECTED_CONFIG)
    print("\nTuning results by best score:")
    print(tuning_results[["stage", "tag", "mean_mcc", "std_mcc"]])
    print("Final selected config:", SELECTED_CONFIG)
    print(f"Elapsed: {time.time() - start:.1f} sec")
else:
    tuning_results = None
    SELECTED_CONFIG = BASE_CONFIG.copy()
    SELECTED_CONFIG["tag"] = config_tag(SELECTED_CONFIG)
    print("RUN_TUNING=False: using default config", SELECTED_CONFIG)

MODEL_DIR = OUTPUT_DIR / "model" / SELECTED_CONFIG["tag"]
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_MODEL_PATH = MODEL_DIR / f"{SELECTED_CONFIG['tag']}.pkl"
FINAL_MODEL_PATH = MODEL_DIR / "final_pipeline.pkl"
SUBMISSION_PATH = MODEL_DIR / "submission.csv"
print("MODEL_DIR:", MODEL_DIR.resolve())
```

### 상세 해설

이 셀은 하이퍼파라미터 튜닝을 수행합니다. `RUN_TUNING=True`이면 여러 후보 설정을 5-fold 교차검증으로 평가하고, 평균 MCC가 가장 높은 설정을 `SELECTED_CONFIG`로 선택합니다.

`evaluate_config` 함수는 하나의 설정을 받아 파이프라인을 만들고 `cross_val_score`로 성능을 측정합니다. `make_scorer(matthews_corrcoef)`로 MCC를 sklearn scorer로 감쌉니다. MCC는 TP, TN, FP, FN을 모두 고려하는 이진 분류 지표이며, 라벨 불균형이 있을 때 accuracy보다 더 신뢰할 수 있습니다. 값은 -1부터 1까지이고 1에 가까울수록 좋습니다.

교차검증은 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`를 사용합니다. Stratified 방식은 각 fold에서 POSITIVE/NEGATIVE 비율을 전체 데이터와 비슷하게 유지합니다. `shuffle=True`와 `random_state=42`는 fold 구성의 재현성을 높입니다.

`cross_val_score`는 각 fold마다 파이프라인 전체를 새로 학습합니다. 중요한 점은 CountVectorizer의 vocabulary와 TfidfTransformer의 IDF도 훈련 fold에서만 학습된다는 것입니다. 이렇게 해야 검증 fold의 정보가 feature 생성 단계에 새어 들어가는 data leakage를 막을 수 있습니다.

튜닝 방식은 전체 그리드서치가 아니라 단계별 탐색입니다. 먼저 세 LinearSVC의 C 조합을 3 x 3 x 3 = 27개 평가합니다. 가장 좋은 SVC 조합을 선택한 뒤, Ridge alpha, LogisticRegression C, ComplementNB alpha, meta LogisticRegression C를 차례대로 평가합니다. 각 단계에서 평균 MCC가 가장 높은 후보를 현재 설정으로 고정하고 다음 단계로 넘어갑니다.

이 방식은 계산량을 줄이기 위한 실용적인 greedy search입니다. 모든 파라미터 조합을 전부 탐색하면 후보 수가 급격히 커지고, 각 후보가 스태킹 모델이라 매우 오래 걸립니다. 단계별 탐색은 전역 최적을 보장하지는 않지만, 교육용 과제와 Colab 환경에서는 현실적인 절충입니다.

`tuning_results`는 모든 후보 평가 결과를 DataFrame으로 모아 평균 MCC 기준 내림차순으로 정렬합니다. 어떤 stage에서 어떤 설정이 좋았는지 확인할 수 있습니다. 튜닝이 끝나면 `SELECTED_CONFIG`와 `MODEL_DIR`, `EXPERIMENT_MODEL_PATH`, `FINAL_MODEL_PATH`, `SUBMISSION_PATH`를 최종 실험 태그 기준으로 다시 설정합니다.

주의할 점은 이 셀이 매우 무겁다는 것입니다. 각 후보마다 5-fold CV를 수행하고, 각 fold 안의 StackingClassifier는 내부적으로 또 3-fold 스태킹 학습을 수행합니다. 로컬 PC에서는 오래 걸릴 수 있으므로 Colab에서 실행하는 것이 적절합니다.

## Code Cell 7 (Notebook Cell 7)

```python
start = time.time()
final_pipeline = build_best_cv_pipeline(SELECTED_CONFIG)
final_pipeline.fit(X, y)

pred_test = final_pipeline.predict(X_test)
submission = pd.DataFrame({
    "row_id": test["row_id"],
    "pred_label": pred_test,
})

assert len(submission) == len(test)
assert set(submission["pred_label"].unique()) <= {"POSITIVE", "NEGATIVE"}

joblib.dump(final_pipeline, EXPERIMENT_MODEL_PATH, compress=3)
joblib.dump(final_pipeline, FINAL_MODEL_PATH, compress=3)
submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8")

print("Saved experiment model:", EXPERIMENT_MODEL_PATH.resolve())
print("Saved final model:", FINAL_MODEL_PATH.resolve())
print("Saved submission:", SUBMISSION_PATH.resolve())
print("Prediction distribution:")
print(submission["pred_label"].value_counts())
print(f"Elapsed: {time.time() - start:.1f} sec")
```

### 상세 해설

이 셀은 선택된 최종 설정으로 전체 학습 데이터를 사용해 모델을 학습하고, 테스트 데이터 예측을 생성한 뒤, 모델 파일과 제출 CSV를 저장합니다. 실제 과제 제출 산출물이 만들어지는 셀입니다.

`final_pipeline = build_best_cv_pipeline(SELECTED_CONFIG)`는 앞 셀에서 선택된 하이퍼파라미터로 완성 파이프라인을 새로 만듭니다. `final_pipeline.fit(X, y)`는 전체 학습 데이터를 사용합니다. 교차검증에서는 검증을 위해 일부 데이터를 제외했지만, 최종 제출 모델은 사용 가능한 모든 학습 데이터를 써야 테스트 예측에 최대한 많은 정보를 활용할 수 있습니다.

fit 과정에서는 세 전처리/벡터화 경로가 각각 학습됩니다. CountVectorizer는 전체 학습 데이터에서 vocabulary를 만들고, TfidfTransformer는 IDF를 계산합니다. 이후 StackingClassifier가 여섯 기본 모델과 최종 LogisticRegression을 학습합니다. 스태킹 내부에서는 기본 모델의 out-of-fold 예측으로 메타 모델을 학습하고, 최종적으로 기본 모델들도 전체 데이터에 다시 학습됩니다.

`pred_test = final_pipeline.predict(X_test)`는 테스트 텍스트에 대해 `POSITIVE` 또는 `NEGATIVE`를 예측합니다. 파이프라인 안에 전처리와 벡터화가 모두 들어 있으므로 테스트 데이터에 별도의 수작업 전처리를 하지 않습니다. 학습 때와 정확히 같은 전처리, vocabulary, IDF, 모델이 적용됩니다.

제출 DataFrame은 `row_id`와 `pred_label` 두 열로 구성합니다. `row_id`는 테스트 파일의 값을 그대로 사용하고, `pred_label`은 모델 예측 결과입니다. 행 수가 테스트 데이터와 같은지, 예측 라벨이 허용된 두 문자열 안에 있는지 assert로 검사합니다.

모델은 두 이름으로 저장됩니다. `EXPERIMENT_MODEL_PATH`는 하이퍼파라미터 태그가 들어간 설명적인 파일명입니다. 나중에 여러 실험을 비교할 때 어떤 설정의 모델인지 추적하기 좋습니다. `FINAL_MODEL_PATH`는 과제 제출에서 요구하는 `final_pipeline.pkl`입니다. 둘 다 같은 최종 파이프라인을 저장합니다.

`submission.to_csv(..., index=False, encoding="utf-8")`는 제출 CSV를 UTF-8로 저장하고 pandas index를 제외합니다. index가 저장되면 `Unnamed: 0` 같은 불필요한 열이 생겨 제출 형식이 틀릴 수 있습니다.

마지막 출력인 prediction distribution은 테스트 예측에서 각 라벨이 몇 번 나왔는지 보여 줍니다. 한쪽 라벨만 극단적으로 많으면 모델 또는 데이터 처리 문제를 의심할 수 있습니다. 성능을 직접 알려 주지는 않지만 제출 전 sanity check로 유용합니다.

## Code Cell 8 (Notebook Cell 8)

```python
check = pd.read_csv(SUBMISSION_PATH, encoding="utf-8")
print(check.shape)
print(check.head())
print(check["pred_label"].value_counts())

assert list(check.columns) == ["row_id", "pred_label"]
assert len(check) == len(test)
assert check["row_id"].equals(test["row_id"])
assert set(check["pred_label"].unique()) <= {"POSITIVE", "NEGATIVE"}
print("submission.csv format OK")
```

### 상세 해설

이 셀은 저장된 `submission.csv`를 다시 읽어서 실제 파일의 제출 형식이 올바른지 검증합니다. 메모리 안의 DataFrame이 아니라 디스크에 저장된 파일을 다시 읽는다는 점이 중요합니다.

`pd.read_csv(SUBMISSION_PATH, encoding="utf-8")`로 제출 파일을 재로드합니다. 저장 중 인코딩 문제가 생겼거나, index가 같이 저장되었거나, 다른 경로에 잘못 저장되었다면 이 단계에서 확인할 수 있습니다.

`check.shape`, `check.head()`, `check["pred_label"].value_counts()`는 파일의 크기, 앞부분 예시, 라벨 분포를 사람이 눈으로 확인하기 위한 출력입니다. 제출 파일은 간단해 보이지만, 실제 대회에서는 형식 오류가 매우 흔하므로 이런 확인이 중요합니다.

첫 번째 assert는 열이 정확히 `row_id`, `pred_label`인지 확인합니다. 두 번째 assert는 제출 행 수가 테스트 행 수와 같은지 확인합니다. 세 번째 assert는 제출 파일의 row_id가 테스트 파일의 row_id와 같은 순서로 정확히 일치하는지 검사합니다. 이 검사는 매우 중요합니다. 행 수가 같아도 row_id 순서가 바뀌면 예측이 엉뚱한 샘플에 매칭될 수 있습니다.

마지막 assert는 예측 라벨이 `POSITIVE`, `NEGATIVE`만 포함하는지 확인합니다. 숫자 라벨, 소문자, 오타, 공백이 섞인 라벨을 방지합니다.

모든 검사를 통과하면 `submission.csv format OK`를 출력합니다. 이 셀은 모델 성능을 올리지는 않지만, 제출 실패를 막는 최종 안전장치입니다. 좋은 모델을 만들었더라도 CSV 형식이 틀리면 점수를 받을 수 없으므로 반드시 필요한 검증 단계입니다.
# 아주 쉬운 설명 버전

이 부분은 위의 자세한 설명을 더 쉽게 다시 풀어쓴 것입니다. 어려운 말이 나오면 바로 쉬운 말로 바꾸어 설명합니다.

## 이 노트북이 하려는 일 한 문장 요약

이 노트북은 사람들이 쓴 한국어 문장을 보고, 그 문장이 좋은 느낌인지 나쁜 느낌인지 맞히는 기계를 만드는 코드입니다.

예를 들어 이런 일을 합니다.

- `정말 좋아요` → `POSITIVE`
- `별로예요` → `NEGATIVE`
- `다시는 안 살래요` → `NEGATIVE`
- `완전 만족합니다` → `POSITIVE`

## 전체 과정을 쉬운 순서로 보기

1. 필요한 도구를 꺼냅니다.
2. 학습용 문제지와 시험지를 찾습니다.
3. 문제지가 제대로 생겼는지 확인합니다.
4. 글자를 깨끗하게 정리하는 청소기를 만듭니다.
5. 문장을 숫자로 바꾸고 여러 판단 기계를 묶은 큰 모델을 만듭니다.
6. 여러 설정을 비교해서 가장 좋은 설정을 고릅니다.
7. 전체 문제지로 최종 공부를 시키고 시험지 답안을 만듭니다.
8. 제출 파일이 형식에 맞는지 다시 확인합니다.

## Code Cell 1 아주 쉬운 설명

첫 번째 셀은 책상 위에 필요한 도구를 모두 꺼내 놓는 단계입니다.

코딩을 할 때는 혼자 모든 일을 직접 만들지 않고, 이미 만들어진 좋은 도구들을 가져와서 씁니다. 여기서는 파일 찾기 도구, 표를 읽는 도구, 글자를 정리하는 도구, 머신러닝 모델을 만드는 도구를 가져옵니다.

`pandas`는 CSV 파일을 표처럼 읽는 도구입니다. 엑셀 표를 다루는 느낌이라고 생각하면 됩니다. `numpy`는 숫자 계산을 빠르게 하는 도구입니다. `sklearn`은 머신러닝 모델을 만들 수 있게 해 주는 도구 상자입니다.

`RANDOM_STATE = 42`는 랜덤으로 섞는 일이 있을 때 매번 똑같이 섞으라는 약속입니다. 요리를 할 때 계량컵을 쓰면 맛이 비슷하게 나오는 것처럼, 머신러닝에서도 랜덤을 고정하면 결과가 더 비슷하게 나옵니다.

이 셀은 아직 모델을 공부시키지 않습니다. 그냥 앞으로 쓸 도구를 준비하는 셀입니다.

## Code Cell 2 아주 쉬운 설명

두 번째 셀은 데이터 파일이 어디 있는지 찾는 단계입니다.

이 노트북은 `public_train.csv`와 `public_test.csv`가 필요합니다. `public_train.csv`는 정답이 있는 연습 문제지이고, `public_test.csv`는 정답 없이 우리가 맞혀야 하는 시험지입니다.

그런데 사람마다 파일을 둔 위치가 다를 수 있습니다. 어떤 사람은 Colab의 Google Drive에 둘 수 있고, 어떤 사람은 컴퓨터의 `mid` 폴더에 둘 수 있습니다. 그래서 이 코드는 여러 장소를 차례대로 뒤져서 두 CSV 파일이 있는 곳을 찾습니다.

찾으면 `DATA_DIR`이라는 이름으로 그 폴더를 기억합니다. 못 찾으면 "어디를 찾아봤는데 없었다"고 자세히 알려 줍니다.

그리고 모델과 제출 파일을 저장할 폴더도 만듭니다. 즉, 이 셀은 집 안에서 문제지와 시험지가 어디 있는지 찾고, 완성된 답안지를 넣을 새 폴더를 만드는 역할입니다.

## Code Cell 3 아주 쉬운 설명

세 번째 셀은 문제지와 시험지를 실제로 열어 보는 단계입니다.

`train`은 정답이 있는 학습 데이터입니다. `test`는 정답이 없는 테스트 데이터입니다.

이 셀은 먼저 데이터가 몇 줄인지, 라벨이 몇 개씩 있는지 봅니다. 그리고 데이터가 약속한 모양인지 검사합니다.

학습 데이터는 반드시 세 칸이 있어야 합니다.

- `row_id`: 문제 번호
- `text`: 문장
- `label`: 정답

테스트 데이터는 두 칸이 있어야 합니다.

- `row_id`: 문제 번호
- `text`: 문장

정답 라벨은 반드시 `POSITIVE`와 `NEGATIVE`만 있어야 합니다. 중간에 `positive`, `1`, `좋음` 같은 다른 이름이 있으면 제출할 때 문제가 됩니다.

마지막으로 `X`, `y`, `X_test`를 만듭니다. 쉽게 말하면 다음과 같습니다.

- `X`: 공부할 문장들
- `y`: 그 문장들의 정답
- `X_test`: 나중에 맞혀야 할 시험 문장들

## Code Cell 4 아주 쉬운 설명

네 번째 셀은 문장을 깨끗하게 정리하는 청소기를 만드는 단계입니다.

사람들이 쓴 문장에는 지저분한 것이 섞일 수 있습니다. 예를 들어 HTML 태그, 인터넷 주소, 이상한 공백, 줄바꿈, 특수문자 같은 것들이 있습니다. 모델이 글을 잘 배우려면 이런 것들을 어느 정도 정리해 주는 것이 좋습니다.

`TextPreprocessor`는 문장을 받아서 깨끗한 문장으로 바꾸는 청소기입니다.

이 청소기는 두 가지 방식으로 일할 수 있습니다.

- `raw`: 띄어쓰기를 유지합니다.
- `nospace`: 띄어쓰기를 없앱니다.

왜 띄어쓰기를 없애는 방식이 필요할까요? 한국어 댓글이나 리뷰에는 띄어쓰기가 틀린 경우가 많기 때문입니다.

예를 들어 아래 세 문장은 사람이 보기에는 비슷합니다.

- `진짜 좋아요`
- `진짜좋아요`
- `진짜  좋아요`

하지만 컴퓨터는 다르게 볼 수 있습니다. 그래서 띄어쓰기를 없앤 버전도 만들어서 컴퓨터가 더 잘 알아보게 합니다.

이 셀은 아직 문장을 숫자로 바꾸지는 않습니다. 문장을 정리하는 방법만 정의합니다.

## Code Cell 5 아주 쉬운 설명

다섯 번째 셀은 진짜 모델의 설계도입니다. 이 노트북에서 가장 중요한 셀입니다.

컴퓨터는 글자를 그대로 이해하지 못합니다. 그래서 문장을 숫자로 바꿔야 합니다. 이 노트북은 문장을 아주 작은 글자 조각으로 쪼갭니다.

예를 들어 `좋아요`라는 말은 이렇게 쪼갤 수 있습니다.

- `좋`
- `아`
- `요`
- `좋아`
- `아요`
- `좋아요`

이런 조각을 `n-gram`이라고 합니다. 쉽게 말하면 글자 조각입니다.

왜 글자 조각을 쓸까요? 감정이 글자 조각에 많이 숨어 있기 때문입니다.

- `좋`은 좋은 느낌일 가능성이 큽니다.
- `별로`는 나쁜 느낌일 가능성이 큽니다.
- `최악`은 나쁜 느낌일 가능성이 큽니다.
- `ㅋㅋ`는 긍정적이거나 가벼운 느낌일 수 있습니다.
- `ㅠㅠ`는 슬픔이나 불만을 나타낼 수 있습니다.

이 셀은 세 가지 방법으로 문장을 숫자로 바꿉니다.

1. 띄어쓰기를 없애고 글자 조각을 봅니다.
2. 띄어쓰기를 유지하고 글자 조각을 봅니다.
3. 단어 단위로도 봅니다.

그 다음 `TF-IDF`라는 방법을 씁니다. 아주 쉽게 말하면, 너무 흔한 말은 점수를 낮추고, 중요한 말은 점수를 높이는 방법입니다. 예를 들어 `그리고`, `입니다` 같은 말은 너무 흔해서 감정을 맞히는 데 별로 도움이 안 될 수 있습니다. 반대로 `최악`, `만족`, `별로` 같은 말은 감정 판단에 더 도움이 될 수 있습니다.

그 다음 여러 판단 기계를 만듭니다.

- LinearSVC 3개
- RidgeClassifier 1개
- LogisticRegression 1개
- ComplementNB 1개

왜 여러 개를 쓸까요? 한 명이 문제를 푸는 것보다 여러 명이 풀고 의견을 모으면 더 잘 맞힐 수 있기 때문입니다. 이 여러 모델을 모아 마지막에 `StackingClassifier`가 최종 결정을 합니다.

`StackingClassifier`는 반장 같은 역할입니다. 여러 모델이 낸 의견을 보고, 마지막으로 어떤 답을 낼지 결정합니다.

결국 이 셀은 "문장을 숫자로 바꾸는 방법"과 "여러 모델이 힘을 합쳐 답을 내는 방법"을 모두 담은 큰 파이프라인을 만드는 셀입니다.

## Code Cell 6 아주 쉬운 설명

여섯 번째 셀은 어떤 설정이 가장 좋은지 시험해 보는 단계입니다.

모델에는 조절할 수 있는 손잡이가 많습니다. 예를 들어 LinearSVC의 `C`, Ridge의 `alpha`, LogisticRegression의 `C`, ComplementNB의 `alpha` 같은 값입니다. 이런 값을 하이퍼파라미터라고 부릅니다. 쉽게 말하면 모델을 만들 때 정하는 조절 버튼입니다.

이 셀은 여러 버튼 값을 바꿔 보면서 어떤 조합이 가장 좋은지 비교합니다.

비교할 때는 `5-fold cross-validation`을 사용합니다. 쉽게 말하면 학습 문제지를 5등분해서, 4개는 공부하고 1개는 시험 보는 일을 5번 반복하는 것입니다. 이렇게 하면 모델이 진짜 잘 배우는지 더 믿을 수 있게 확인할 수 있습니다.

점수는 `MCC`로 계산합니다. MCC는 맞힌 것과 틀린 것을 균형 있게 보는 점수입니다. 한쪽 라벨만 많이 맞혀서 좋아 보이는 일을 막아 줍니다.

이 셀은 시간이 오래 걸릴 수 있습니다. 왜냐하면 여러 설정을 각각 5번씩 시험하고, 모델 안에서도 여러 모델이 함께 학습하기 때문입니다. 그래서 로컬 컴퓨터보다 Colab에서 돌리는 것이 더 좋을 수 있습니다.

마지막에는 가장 점수가 좋은 설정을 `SELECTED_CONFIG`로 저장합니다. 다음 셀은 이 설정으로 최종 모델을 만듭니다.

## Code Cell 7 아주 쉬운 설명

일곱 번째 셀은 최종 모델을 진짜로 공부시키고 제출 파일을 만드는 단계입니다.

앞에서는 여러 설정을 비교하느라 학습 데이터를 나누어 썼습니다. 하지만 이제 최종 제출을 해야 하므로, 정답이 있는 모든 학습 데이터를 다 사용해서 모델을 공부시킵니다.

`final_pipeline.fit(X, y)`는 모델에게 모든 학습 문장과 정답을 보여 주는 일입니다. 모델은 이 과정에서 어떤 글자 조각이 긍정과 관련 있고, 어떤 글자 조각이 부정과 관련 있는지 배웁니다.

그 다음 `final_pipeline.predict(X_test)`로 테스트 문장의 정답을 예측합니다. 결과는 `POSITIVE` 또는 `NEGATIVE`입니다.

그 후 제출 파일을 만듭니다. 제출 파일에는 딱 두 칸이 있어야 합니다.

- `row_id`: 테스트 문제 번호
- `pred_label`: 모델이 예측한 정답

그리고 모델도 저장합니다. 하나는 실험 이름이 들어간 파일이고, 하나는 과제 제출용 이름인 `final_pipeline.pkl`입니다.

쉽게 말하면 이 셀은 공부를 끝낸 모델과 답안지를 저장하는 단계입니다.

## Code Cell 8 아주 쉬운 설명

여덟 번째 셀은 제출 파일이 제대로 만들어졌는지 마지막으로 확인하는 단계입니다.

저장한 `submission.csv`를 다시 열어 봅니다. 왜 다시 열까요? 저장할 때 실수로 이상한 열이 들어갔거나, 행 수가 틀렸거나, row_id 순서가 바뀌었을 수 있기 때문입니다.

이 셀은 다음을 확인합니다.

- 열 이름이 정확히 `row_id`, `pred_label`인지
- 행 수가 테스트 데이터와 같은지
- row_id 순서가 테스트 데이터와 같은지
- 예측 라벨이 `POSITIVE`, `NEGATIVE`만 있는지

모두 통과하면 `submission.csv format OK`가 나옵니다.

이 셀은 모델을 더 똑똑하게 만들지는 않습니다. 하지만 제출 실수를 막아 줍니다. 아무리 모델이 좋아도 제출 파일 형식이 틀리면 점수를 못 받을 수 있으므로 아주 중요한 마지막 점검입니다.

## 핵심 용어 아주 쉽게 정리

### CSV

표를 저장한 파일입니다. 엑셀 표와 비슷하다고 생각하면 됩니다.

### train 데이터

정답이 있는 연습 문제지입니다. 모델이 이것을 보고 공부합니다.

### test 데이터

정답이 없는 시험지입니다. 모델이 답을 맞혀야 합니다.

### label

정답입니다. 이 노트북에서는 `POSITIVE` 또는 `NEGATIVE`입니다.

### POSITIVE

좋은 느낌이라는 뜻입니다.

### NEGATIVE

나쁜 느낌이라는 뜻입니다.

### 전처리

모델이 보기 좋게 문장을 깨끗하게 정리하는 일입니다.

### n-gram

문장을 작은 글자 조각이나 단어 조각으로 나눈 것입니다.

### CountVectorizer

문장을 숫자로 바꾸는 도구입니다. 어떤 글자 조각이 나왔는지 표시합니다.

### TF-IDF

흔한 말은 덜 중요하게 보고, 특별하고 구분에 도움이 되는 말은 더 중요하게 보는 방법입니다.

### Pipeline

여러 작업을 순서대로 묶은 기계입니다. 문장 정리, 숫자 변환, 예측을 한 번에 처리합니다.

### LinearSVC, RidgeClassifier, LogisticRegression, ComplementNB

각각 다른 방식으로 문장이 긍정인지 부정인지 판단하는 모델입니다. 이 노트북은 이 모델들을 여러 개 같이 씁니다.

### StackingClassifier

여러 모델의 의견을 모아서 최종 답을 정하는 모델입니다. 여러 친구가 낸 답을 보고 반장이 최종 답을 고르는 것과 비슷합니다.

### Cross-validation

학습 데이터를 여러 조각으로 나누어 공부와 시험을 번갈아 해 보는 방법입니다. 모델이 진짜 잘하는지 확인하는 데 씁니다.

### MCC

분류 모델의 점수입니다. 긍정과 부정을 균형 있게 잘 맞히는지 봅니다.

### final_pipeline.pkl

최종 모델을 저장한 파일입니다. 나중에 다시 불러와서 예측할 수 있습니다.

### submission.csv

리더보드나 과제 시스템에 제출하는 답안지 파일입니다.

## 마지막으로 정말 쉽게 말하면

이 노트북은 한국어 문장을 많이 읽고, 어떤 글자 조각이 좋은 느낌인지 나쁜 느낌인지 배웁니다. 그리고 여러 판단 모델이 함께 의견을 낸 뒤, 마지막 모델이 최종 답을 정합니다. 마지막에는 시험 데이터의 답안지 `submission.csv`와 다시 사용할 수 있는 모델 파일 `final_pipeline.pkl`을 저장합니다.
