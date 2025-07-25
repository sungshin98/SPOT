import os

# 현재 디렉토리 기준 metadata.csv 경로 정의
BASE_DIR = os.path.dirname(__file__)
metadata_csv = os.path.join(BASE_DIR, "metadata.csv")

__all__= ['metadata_csv']
