import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px # 인터랙티브 시각화 라이브러리
import seaborn as sns
from sklearn.manifold import TSNE
from pykeen.pipeline import pipeline
from sklearn.decomposition import PCA
import pykeen.predict
from pykeen.datasets import Nations # 샘플 데이터셋 추후 민석이 구해온 데이터셋으로 대체 예정


OUTPUT = 'transe_result'
os.makedirs(OUTPUT, exist_ok=True) # OUTPUT 디렉터리생성

device = torch.device("cpu")

print("TrasE 모델 학습 및 시각화 시작")

# pipeline 함수로 모델 학습 및 평가를 수행(자동으로 파라미터 설정해줌)
result = pipeline(
    dataset='nations',
    model='TransE',
    # 모델 설정
    model_kwargs=dict(
        embedding_dim=50, # 임베딩 차원
        scoring_fct_norm=1, # L1 거리 공식 사용
    ),
    device=device,
    # 학습 설정
    training_kwargs=dict(
        num_epochs=100, # 에폭 수
        batch_size=64, # 배치 크기
        use_tqdm=True,
    ),
    # 정규화 설정
    optimizer_kwargs=dict(lr=1e-2),
)

# 학습 결과 저장
result.save_to_directory(OUTPUT)
print(f"모델과 결과를 '{OUTPUT}'에 저장.")

# 모델 성능평가(MRR, Hits@1, Hits@10)
# MetricResults 객체에서 주요 지표를 가져옵니다 (Filtered 방식)
print("\n--- 모델 성능 평가 ---")
results_dict = result.metric_results.to_flat_dict()
mrr = results_dict.get('both.realistic.inverse_harmonic_mean_rank')

hits_at_10 = results_dict.get('both.realistic.hits_at_10') or \
             results_dict.get('both.avg.hits_at_10')

print(f"MRR: {mrr:.4f}" if mrr else "MRR 키를 찾을 수 없음.")
print(f"Hits@10: {hits_at_10:.4f}" if hits_at_10 else "Hits@10 키를 찾을 수 없음.")


# 시각화를 위한 임베딩 추출 및 차원 축소
print("\n시각화 준비. 임베딩 추출 및 차원 축소 실행")

# 학습된 모델
model = result.model

# 팩토리를 result 객체에서 직접 가져오기
training_factory = result.training

# 모든 개체의 이름과 해당 벡터 가져오기
# PyKeen에서 ID와 이름을 매핑 해준다
entity_to_id = training_factory.entity_to_id
entity_names = list(entity_to_id.keys()) # 개체 이름 리스트
entity_ids = torch.tensor(list(training_factory.entity_to_id.values()), device=model.device) # 개체 ID 텐서
# 개체 임베딩 벡터 가져오기 (GPU에서 CPU로 이동 후 NumPy 배열로 변환)
entity_embeddings = model.entity_representations[0](entity_ids).detach().cpu().numpy()

# 관계 이름과 벡터 가져오기
relation_to_id = training_factory.relation_to_id
relation_names = list(relation_to_id.keys()) # 관계 이름 리스트
relation_ids = torch.tensor(list(relation_to_id.values()), device=model.device) # 관계 ID 텐서
# 관계 임베딩 벡터 가져오기 (GPU에서 CPU로 이동 후 NumPy 배열로 변환)
relation_embeddings = model.relation_representations[0](relation_ids).detach().cpu().numpy()

# --- 2차원 시각화 ---
# # 고차원을 2차원으로 축소하기 위해 t-SNE 객체 생성
# tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(entity_names)-1), max_iter=1000)
# print("차원 축소 중...")
# # tsne로 entity_embeddings 학습 및 2차원으로 변환, X_embedded는 (x,y) 형태의 배열
# X_embedded = tsne.fit_transform(entity_embeddings)

# # 시각화를 위한 DataFrame 생성
# df_tsne = pd.DataFrame(X_embedded, columns=['x', 'y'])
# # 개체 이름을 매핑
# df_tsne['entity'] = entity_names

# # Nations 데이터셋은 나라 이름이므로 앞글자로 임의 분류
# df_tsne['category'] = df_tsne['entity'].apply(lambda x: x[0].upper())

# # 시각화 matplotlib와 seaborn 사용
# print("시각화 중...")
# plt.figure(figsize=(12, 10))
# sns.set_style("whitegrid")

# # 카테고리별 산점도 그리기
# scatter = sns.scatterplot(
#     data=df_tsne,
#     x='x', y='y',
#     hue='category',
#     palette='viridis',
#     s=100, # 점 크기
#     alpha=0.8,
#     legend='full'
# )

# # 점 위에 개체 이름 표시
# for i in range(df_tsne.shape[0]):
#     plt.text(
#         x=df_tsne.x[i]+0.2, 
#         y=df_tsne.y[i]+0.2, 
#         s=df_tsne.entity[i], 
#         fontdict=dict(color='black', size=9),
#         alpha=0.7
#     )

# plt.title('TransE embedding entity vectors (TSNE)', fontsize=15)
# plt.xlabel('d 1', fontsize=12)
# plt.ylabel('d 2', fontsize=12)
# plt.legend(title='Category', loc='best')

# # 결과 이미지 저장
# plt.savefig(os.path.join(OUTPUT, 'transe_tsne_visualization.png'), dpi=300)
# print(f"Visualization plot saved to '{OUTPUT}/transe_tsne_visualization.png'")

# plt.show()

# # 관계 벡터 시각화 (관계는 수가 적으므로 PCA로 경향성만 확인)
# if len(relation_names) > 2:
#     print("\n관계 벡터 시각화 (PCA 사용)")
#     pca = PCA(n_components=2)
#     rel_embedded = pca.fit_transform(relation_embeddings)
#     df_rel_pca = pd.DataFrame(rel_embedded, columns=['x', 'y'])
#     df_rel_pca['relation'] = relation_names

#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(data=df_rel_pca, x='x', y='y', s=100, color='red')
#     for i in range(df_rel_pca.shape[0]):
#         plt.text(df_rel_pca.x[i]+0.01, df_rel_pca.y[i]+0.01, df_rel_pca.relation[i], size=9)
#     plt.title('TransE embedding relation vectors (PCA)', fontsize=15)
#     plt.savefig(os.path.join(OUTPUT, 'transe_relation_pca.png'))
#     plt.show()

# --- 3차원 시각화 (Plotly 사용) ---
print("3차원 차원 축소 중...")
tsne_3d = TSNE(
    n_components=3, 
    random_state=42, 
    perplexity=min(30, len(entity_names)-1),
    max_iter=1000
)
X_3d = tsne_3d.fit_transform(entity_embeddings)

# DataFrame 생성 및 개체 이름 매핑
df_3d = pd.DataFrame(X_3d, columns=['x', 'y', 'z'])
df_3d['entity'] = entity_names

# 3. Plotly를 이용한 3D 산점도 생성
fig = px.scatter_3d(
    df_3d, 
    x='x', y='y', z='z',
    text='entity',      # 점 위에 마우스를 올리면 이름이 나옴
    color='x',          # 좌표값에 따라 색상을 입혀 입체감을 높임
    title='TransE Entity Embeddings (3D t-SNE)',
    labels={'x': 'TSNE-1', 'y': 'TSNE-2', 'z': 'TSNE-3'}
)

# 4. 점 크기 및 라벨 설정
fig.update_traces(marker=dict(size=5))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

# 5. 그래프 출력 및 HTML 저장
fig.show()
fig.write_html(os.path.join(OUTPUT, 'transe_3d_visualization.html'))
print(f"3D 시각화 완료: {OUTPUT}/transe_3d_visualization.html")

# --- 링크 예측 테스트 (Link Prediction) ---
print("\n--- 링크 예측 테스트 (Link Prediction) ---")

# 데이터셋에 들어있는 실제 관계 이름들 확인 
relations = list(result.training.relation_to_id.keys())
# 'diplomatic'이 포함된 실제 관계 이름을 찾기
target_rel = [r for r in relations if 'diplom' in r.lower()][0] 
print(f"예측에 사용할 실제 관계 이름: {target_rel}")

# 2. 찾은 정확한 이름을 넣어서 예측 실행
df_tail = pykeen.predict.predict_target(
    model=result.model,
    head="brazil",
    relation=target_rel, # 'diplomatic' 대신 찾은 변수 사용
    triples_factory=result.training,
).df

# 결과 상위 10개 출력
print(df_tail.head(10))