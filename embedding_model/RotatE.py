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
from pykeen.datasets import FB15k237 # 샘플 데이터셋 추후 민석이 구해온 데이터셋으로 대체 예정


OUTPUT = 'rotatee_result'
os.makedirs(OUTPUT, exist_ok=True) # OUTPUT 디렉터리생성

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 장치: {device}")

print("RotateE 모델 학습 및 시각화 시작")

# pipeline 함수로 모델 학습 및 평가를 수행(자동으로 파라미터 설정해줌)
result = pipeline(
    dataset='FB15k237',
    model='Rotate',
    # 모델 설정
    model_kwargs=dict(
        embedding_dim=500, # 임베딩 차원
    ),
    device=device,
    # 학습 설정
    training_kwargs=dict(
        num_epochs=700, # 에폭 수
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
complex_embeddings = model.entity_representations[0](entity_ids).detach().cpu().numpy()

# 복소수를 실수로 변환 (Real part와 Imaginary part를 결합)
# 복소수 [a + bi]를 [a, b] 형태의 실수 벡터로 변환
real_part = np.real(complex_embeddings)
imag_part = np.imag(complex_embeddings)
entity_embeddings = np.concatenate([real_part, imag_part], axis=-1)

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

# plt.title('RotateE embedding entity vectors (TSNE)', fontsize=15)
# plt.xlabel('d 1', fontsize=12)
# plt.ylabel('d 2', fontsize=12)
# plt.legend(title='Category', loc='best')

# # 결과 이미지 저장
# plt.savefig(os.path.join(OUTPUT, 'rotatee_tsne_visualization.png'), dpi=300)
# print(f"Visualization plot saved to '{OUTPUT}/rotatee_tsne_visualization.png'")

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
#     plt.title('RotateE embedding relation vectors (PCA)', fontsize=15)
#     plt.savefig(os.path.join(OUTPUT, 'rotatee_relation_pca.png'))
#     plt.show()

# 3. 3D 시각화 (Plotly 사용) - 샘플링 추가
# -------------------------------------------------------------
print("3차원 차원 축소 중...")
# FB15k237은 14000개가 넘으므로 시각화 시 브라우저 과부하 방지를 위해 랜덤 샘플링
SAMPLE_SIZE = min(1000, len(entity_names))
np.random.seed(42)
sample_indices = np.random.choice(len(entity_names), SAMPLE_SIZE, replace=False)

sampled_embeddings = entity_embeddings[sample_indices]
sampled_entity_names = [entity_names[i] for i in sample_indices]

tsne_3d = TSNE(
    n_components=3,
    random_state=42,
    perplexity=min(30, len(sampled_entity_names)-1),
    max_iter=1000
)
X_3d = tsne_3d.fit_transform(sampled_embeddings)

# DataFrame 생성 및 개체 이름 매핑
df_3d = pd.DataFrame(X_3d, columns=['x', 'y', 'z'])
df_3d['entity'] = sampled_entity_names

# Plotly를 이용한 3D 산점도 생성
fig = px.scatter_3d(
    df_3d,
    x='x', y='y', z='z',
    text='entity',
    color='x',
    title='RotateE Entity Embeddings (3D t-SNE) - Sampled',
    labels={'x': 'TSNE-1', 'y': 'TSNE-2', 'z': 'TSNE-3'}
)

fig.update_traces(marker=dict(size=5), textposition='top center')
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

fig.write_html(os.path.join(OUTPUT, 'rotatee_3d_visualization.html'))
print(f"3D 시각화 완료: {OUTPUT}/rotatee_3d_visualization.html")

# --- 링크 예측 테스트 (Link Prediction) ---
print("\n--- 링크 예측 테스트 (Link Prediction) ---")

# 데이터셋에 들어있는 실제 관계 이름들 확인 
relations = list(result.training.relation_to_id.keys())
# 'diplomatic'이 포함된 실제 관계 이름을 찾기
target_rel = [r for r in relations if 'diplom' in r.lower()][0] 
print(f"예측에 사용할 실제 관계 이름: {target_rel}")

# --- 링크 예측 테스트 (Link Prediction) ---
print("\n--- 링크 예측 테스트 (Link Prediction) ---")

# 데이터셋에 들어있는 실제 관계 이름들 확인
relations = list(result.training.relation_to_id.keys())
# 'diplom'이 포함된 관계가 있는지 찾고, 없으면 데이터셋의 첫 번째 관계를 사용
diplom_rels = [r for r in relations if 'diplom' in r.lower()]
target_rel = diplom_rels[0] if diplom_rels else relations[0]
print(f"예측에 사용할 실제 관계 이름: {target_rel}")

# 2. 찾은 정확한 이름을 넣어서 예측 실행
# 'brazil' 엔티티가 있는지 확인하고, 없으면 데이터셋의 첫 번째 엔티티 사용
target_head = "brazil" if "brazil" in entity_names else entity_names[0]
print(f"예측에 사용할 Head 엔티티: {target_head}")

# 예측 실행
df_tail = pykeen.predict.predict_target(
    model=result.model,
    head=target_head,
    relation=target_rel,
    triples_factory=result.training,
).df

# 결과 상위 10개 출력
print(f"\n[{target_head}] ---({target_rel})---> [?] 에 대한 꼬리(Tail) 예측 결과:")
print(df_tail.head(10))