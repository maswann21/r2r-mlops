# Project Restructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 프로젝트 폴더 구조를 단순화 — 모든 이미지 데이터를 `data/`로 통합, 미사용 stub/sensor 코드 제거, 루트 레벨 파일 정리.

**Architecture:** 파일 이동/삭제 위주의 구조 정리. 코드 변경 없음. 각 Task 후 `git status`로 확인 후 커밋.

**Tech Stack:** bash, git

---

## Task 1: 데이터 디렉토리 생성 및 classification 이동

**Files:**
- Create: `data/classification/`
- Move: `db/classification/images/` → `data/classification/images/`
- Move: `db/classification/labels.csv` → `data/classification/labels.csv`
- Delete: `db/classification/` (빈 폴더)

**Step 1: data/ 디렉토리 생성**

```bash
mkdir -p /mnt/c/Users/user/r2r/r2r-mlops/data/classification
```

Expected: 오류 없음

**Step 2: classification 이미지 및 라벨 이동**

```bash
mv /mnt/c/Users/user/r2r/r2r-mlops/db/classification/images \
   /mnt/c/Users/user/r2r/r2r-mlops/data/classification/images

mv /mnt/c/Users/user/r2r/r2r-mlops/db/classification/labels.csv \
   /mnt/c/Users/user/r2r/r2r-mlops/data/classification/labels.csv
```

**Step 3: 빈 폴더 제거**

```bash
rmdir /mnt/c/Users/user/r2r/r2r-mlops/db/classification
```

**Step 4: 이동 확인**

```bash
ls /mnt/c/Users/user/r2r/r2r-mlops/data/classification/
ls /mnt/c/Users/user/r2r/r2r-mlops/data/classification/images/ | wc -l
```

Expected:
```
images  labels.csv
1010
```

**Step 5: Commit**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
git add -A
git commit -m "refactor: move db/classification to data/classification"
```

---

## Task 2: detection 데이터 이동

**Files:**
- Create: `data/detection/`
- Move: `db/detection/images/` → `data/detection/images/`
- Move: `db/detection/labels/` → `data/detection/labels/`
- Delete: `db/detection/` (빈 폴더)

**Step 1: detection 디렉토리 이동**

```bash
mkdir -p /mnt/c/Users/user/r2r/r2r-mlops/data/detection

mv /mnt/c/Users/user/r2r/r2r-mlops/db/detection/images \
   /mnt/c/Users/user/r2r/r2r-mlops/data/detection/images

mv /mnt/c/Users/user/r2r/r2r-mlops/db/detection/labels \
   /mnt/c/Users/user/r2r/r2r-mlops/data/detection/labels
```

**Step 2: 빈 폴더 제거**

```bash
rmdir /mnt/c/Users/user/r2r/r2r-mlops/db/detection
```

**Step 3: 확인**

```bash
ls /mnt/c/Users/user/r2r/r2r-mlops/data/detection/
ls /mnt/c/Users/user/r2r/r2r-mlops/data/detection/images/ | wc -l
```

Expected:
```
images  labels
2227
```

**Step 4: Commit**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
git add -A
git commit -m "refactor: move db/detection to data/detection"
```

---

## Task 3: segmentation 데이터 이동

**Files:**
- Create: `data/segmentation/`
- Move: `db/segmentation/images/` → `data/segmentation/images/`
- Move: `db/segmentation/masks/` → `data/segmentation/masks/`
- Delete: `db/segmentation/` (빈 폴더)

**Step 1: segmentation 디렉토리 이동**

```bash
mkdir -p /mnt/c/Users/user/r2r/r2r-mlops/data/segmentation

mv /mnt/c/Users/user/r2r/r2r-mlops/db/segmentation/images \
   /mnt/c/Users/user/r2r/r2r-mlops/data/segmentation/images

mv /mnt/c/Users/user/r2r/r2r-mlops/db/segmentation/masks \
   /mnt/c/Users/user/r2r/r2r-mlops/data/segmentation/masks
```

**Step 2: 빈 폴더 제거**

```bash
rmdir /mnt/c/Users/user/r2r/r2r-mlops/db/segmentation
```

**Step 3: 확인**

```bash
ls /mnt/c/Users/user/r2r/r2r-mlops/data/segmentation/
ls /mnt/c/Users/user/r2r/r2r-mlops/data/segmentation/images/ | wc -l
```

Expected:
```
images  masks
2227
```

**Step 4: Commit**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
git add -A
git commit -m "refactor: move db/segmentation to data/segmentation"
```

---

## Task 4: 루트 레벨 파일 정리

**Files:**
- Move: `data.csv` → `data/raw/data.csv`
- Move: `WBS.md` → `docs/WBS.md`
- Move: `R2R Coating 불량 감지 및 자동 최적화 MLOps 프로젝트설명.md` → `docs/project-overview.md`

**Step 1: data/raw 생성 및 data.csv 이동**

```bash
mkdir -p /mnt/c/Users/user/r2r/r2r-mlops/data/raw

mv "/mnt/c/Users/user/r2r/r2r-mlops/data.csv" \
   /mnt/c/Users/user/r2r/r2r-mlops/data/raw/data.csv
```

**Step 2: 문서 파일 이동**

```bash
mv /mnt/c/Users/user/r2r/r2r-mlops/WBS.md \
   /mnt/c/Users/user/r2r/r2r-mlops/docs/WBS.md

mv "/mnt/c/Users/user/r2r/r2r-mlops/R2R Coating 불량 감지 및 자동 최적화 MLOps 프로젝트설명.md" \
   /mnt/c/Users/user/r2r/r2r-mlops/docs/project-overview.md
```

**Step 3: 루트 확인 (잡파일 없어야 함)**

```bash
ls /mnt/c/Users/user/r2r/r2r-mlops/*.md /mnt/c/Users/user/r2r/r2r-mlops/*.csv 2>/dev/null
```

Expected: `CLAUDE.md  README.md` 만 출력 (data.csv, WBS.md 없어야 함)

**Step 4: Commit**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
git add -A
git commit -m "refactor: move root-level loose files to data/ and docs/"
```

---

## Task 5: 빈 stub 모듈 제거

**Files:**
- Delete: `src/__init__.py`
- Delete: `src/data/`
- Delete: `src/features/`
- Delete: `src/training/`
- Delete: `src/utils/`
- Delete: `src/models/__init__.py`

**Step 1: 빈 stub 디렉토리 제거**

```bash
rm -rf \
  /mnt/c/Users/user/r2r/r2r-mlops/src/data \
  /mnt/c/Users/user/r2r/r2r-mlops/src/features \
  /mnt/c/Users/user/r2r/r2r-mlops/src/training \
  /mnt/c/Users/user/r2r/r2r-mlops/src/utils \
  /mnt/c/Users/user/r2r/r2r-mlops/src/__init__.py \
  /mnt/c/Users/user/r2r/r2r-mlops/src/models/__init__.py
```

**Step 2: src/ 구조 확인**

```bash
find /mnt/c/Users/user/r2r/r2r-mlops/src -type f
```

Expected:
```
src/models/vision/__init__.py
src/models/vision/classification.py
```

**Step 3: Commit**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
git add -A
git commit -m "refactor: remove empty stub modules from src/"
```

---

## Task 6: sensor 코드 및 ml/ 정리

**Files:**
- Delete: `src/models/sensor/`
- Delete: `ml/models/sensor/`
- Delete: `ml/.gitkeep`
- Move: `ml/Machine_Failure_Prediction_using_Sensor_data.ipynb` → `notebooks/sensor/`

**Step 1: sensor 소스코드 제거**

```bash
rm -rf /mnt/c/Users/user/r2r/r2r-mlops/src/models/sensor
```

**Step 2: ml/models/sensor/ 빈 폴더 제거**

```bash
rm -rf /mnt/c/Users/user/r2r/r2r-mlops/ml/models/sensor
rm -f /mnt/c/Users/user/r2r/r2r-mlops/ml/.gitkeep
```

**Step 3: 노트북 이동**

```bash
mv "/mnt/c/Users/user/r2r/r2r-mlops/ml/Machine_Failure_Prediction_using_Sensor_data.ipynb" \
   /mnt/c/Users/user/r2r/r2r-mlops/notebooks/sensor/
```

**Step 4: ml/ 및 src/models/ 확인**

```bash
find /mnt/c/Users/user/r2r/r2r-mlops/ml -type f
find /mnt/c/Users/user/r2r/r2r-mlops/src/models -type f
```

Expected:
```
ml/models/vision/classification/resnet18_defect.pth
src/models/vision/__init__.py
src/models/vision/classification.py
```

**Step 5: Commit**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
git add -A
git commit -m "refactor: remove sensor code and clean up ml/ directory"
```

---

## Task 7: 설정 파일 및 문서 경로 업데이트

**Files:**
- Modify: `config/config.yaml`
- Modify: `setup.py`
- Modify: `CLAUDE.md`
- Modify: `docs/PROJECT_STRUCTURE.md`

**Step 1: config/config.yaml — vision 데이터 경로 추가**

`config/config.yaml`의 `datasets:` 섹션 아래에 다음을 추가:

```yaml
  - name: "vision_classification"
    path: "data/classification/"
    type: "Directory"
```

**Step 2: setup.py — entry_points의 sensor 참조 제거**

`setup.py`의 `entry_points`를 아래와 같이 수정:

```python
entry_points={
    "console_scripts": [
        "r2r-api=api.main:main",
    ],
},
```

(`r2r-train=src.training.train:main` 줄 제거 — `src/training/` 삭제했으므로)

**Step 3: CLAUDE.md — 데이터 경로 참조 업데이트**

CLAUDE.md의 Vision Data 섹션을 확인해 `db/classification` 참조를 `data/classification`으로 수정.

**Step 4: docs/PROJECT_STRUCTURE.md — 구조도 업데이트**

`data/` 섹션 추가, `db/` 섹션에서 이미지 데이터 제거.

**Step 5: 확인**

```bash
grep -r "db/classification\|db/detection\|db/segmentation" \
  /mnt/c/Users/user/r2r/r2r-mlops/config \
  /mnt/c/Users/user/r2r/r2r-mlops/CLAUDE.md \
  /mnt/c/Users/user/r2r/r2r-mlops/docs
```

Expected: 출력 없음 (참조 없어야 함)

**Step 6: Commit**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
git add -A
git commit -m "refactor: update config and docs to reflect new data/ paths"
```

---

## Task 8: 최종 검증

**Step 1: 전체 구조 확인**

```bash
find /mnt/c/Users/user/r2r/r2r-mlops \
  -not -path '*/.git/*' \
  -not -path '*/data/classification/images/*' \
  -not -path '*/data/detection/images/*' \
  -not -path '*/data/segmentation/images/*' \
  -not -path '*/data/segmentation/masks/*' \
  | sort
```

Expected 최종 구조:
```
r2r-mlops/
├── api/__init__.py
├── config/config.yaml, params.yaml
├── data/
│   ├── classification/images/ + labels.csv
│   ├── detection/images/ + labels/
│   ├── raw/data.csv
│   └── segmentation/images/ + masks/
├── db/schema.sql
├── docker/
├── docs/
├── ml/models/vision/classification/resnet18_defect.pth
├── notebooks/sensor/*.ipynb, vision/
├── src/models/vision/__init__.py + classification.py
├── CLAUDE.md, README.md, requirements.txt, setup.py
```

**Step 2: classification.py import 정상 확인**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
python -c "from src.models.vision import ResNetClassifier; print('OK')"
```

Expected: `OK`

**Step 3: git log 확인**

```bash
git log --oneline -8
```

Expected: Task 1~7의 커밋 8개 표시
