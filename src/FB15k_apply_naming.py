import pandas as pd
import os

# 1. 경로 설정
RESULT_FILE = 'rotate_result/sampled_top1_results.csv'
MAPPING_FILE = 'FB15k_mid2name.txt' 
OUTPUT_FILE = 'rotate_result/final_human_readable_results.csv'

def main():
    if not os.path.exists(MAPPING_FILE):
        print(f" 매핑 파일('{MAPPING_FILE}')을 찾을 수 없습니다.")
        return

    print("데이터 로딩 중...")
    # 2. 결과 CSV 로드
    df = pd.read_csv(RESULT_FILE)

    # 3. mid2name 매핑 딕셔너리 생성
    # FB15k_mid2name.txt 파일의 형식을 확인하고 읽어옵니다.
    mapping = {}
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # /m/0xxx -> 실제 이름
                mapping[parts[0]] = parts[1]

    print(f"매핑 사전 구축 완료: {len(mapping)}개 엔티티")

    # 4. 이름 변환 적용
    # 데이터셋에 존재하지 않는 ID일 경우 원래 ID를 유지하도록 .get() 사용
    df['head_name'] = df['head'].apply(lambda x: mapping.get(x, x))
    df['tail_name'] = df['tail'].apply(lambda x: mapping.get(x, x))

    # 5. 컬럼 정리 및 저장
    # 가독성을 위해 이름 컬럼을 앞으로 배치합니다.
    final_df = df[['head_name', 'relation', 'tail_name', 'score']]
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"✨ 작업 완료! 결과 파일: {OUTPUT_FILE}")
    print("\n--- 실제 이름으로 변환된 추론 샘플 ---")
    print(final_df.head(10))

if __name__ == "__main__":
    main()