import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px  # 인터랙티브 시각화 라이브러리
import seaborn as sns
from sklearn.manifold import TSNE
from pykeen.pipeline import pipeline
from sklearn.decomposition import PCA
import pykeen.predict
from pykeen.datasets import FB15k237

# 구글 코랩 로컬 세션에 저장 디렉터리 생성
OUTPUT = 'complexe_result'
os.makedirs(OUTPUT, exist_ok=True) 

# 코랩 GPU 장치 연결 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 장치: {device}")
if torch.cuda.is_available():
    print(f"연결된 GPU: {torch.cuda.get_device_name(0)}")

print("ComplEx 모델 학습 및 시각화 시작")

# pipeline 함수로 모델 학습 및 평가를 수행
result = pipeline(
    dataset='FB15k237', 
    model='ComplEx',
    model_kwargs=dict(embedding_dim=500),
    device=device,
    training_kwargs=dict(
        num_epochs=700,
        batch_size=64, 
        use_tqdm=True,
    ),
    optimizer_kwargs=dict(lr=1e-2),
    
    # ── Early Stopper 기능 반영 ──────────────────────────
    stopper='early',
    stopper_kwargs=dict(
        frequency=25,       # 25에포크마다 검증(Evaluation) 수행
        patience=5,         # 성능이 연속 5번(125에포크 동안) 개선되지 않으면 종료
        relative_delta=0.0001,
        metric='mrr',       
    ),
    evaluator_kwargs=dict(filtered=True),
    # 코랩 환경에서도 검증 시 OOM(메모리 부족) 에러 방지를 위해 CPU 할당
    evaluation_kwargs=dict(
        batch_size=16,
        device=torch.device('cpu')
    ),
)

# 학습 결과 저장
result.save_to_directory(OUTPUT)
print(f"모델과 결과를 '{OUTPUT}'에 저장.")

# 모델 성능평가(MRR, Hits@1, Hits@10)
print("\n--- 모델 성능 평가 ---")
results_dict = result.metric_results.to_flat_dict()
mrr = results_dict.get('both.realistic.inverse_harmonic_mean_rank')
hits_at_10 = results_dict.get('both.realistic.hits_at_10') or results_dict.get('both.avg.hits_at_10')

print(f"MRR: {mrr:.4f}" if mrr else "MRR 키를 찾을 수 없음.")
print(f"Hits@10: {hits_at_10:.4f}" if hits_at_10 else "Hits@10 키를 찾을 수 없음.")


# 시각화를 위한 임베딩 추출 및 차원 축소
print("\n시각화 준비. 임베딩 추출 및 차원 축소 실행")
model = result.model
training_factory = result.training

entity_to_id = training_factory.entity_to_id
entity_names = list(entity_to_id.keys()) 
entity_ids = torch.tensor(list(training_factory.entity_to_id.values()), device=model.device) 
complex_embeddings = model.entity_representations[0](entity_ids).detach().cpu().numpy()

# 복소수를 실수로 변환
real_part = np.real(complex_embeddings)
imag_part = np.imag(complex_embeddings)
entity_embeddings = np.concatenate([real_part, imag_part], axis=-1)

# 3차원 차원 축소
print("3차원 차원 축소 중...")
tsne_3d = TSNE(
    n_components=3, 
    random_state=42, 
    perplexity=min(30, len(entity_names)-1),
    max_iter=1000
)
X_3d = tsne_3d.fit_transform(entity_embeddings)

df_3d = pd.DataFrame(X_3d, columns=['x', 'y', 'z'])
df_3d['entity'] = entity_names

fig = px.scatter_3d(
    df_3d, 
    x='x', y='y', z='z',
    text='entity',      
    color='x',          
    title='ComplEx Entity Embeddings (3D t-SNE)',
    labels={'x': 'TSNE-1', 'y': 'TSNE-2', 'z': 'TSNE-3'}
)

fig.update_traces(marker=dict(size=5))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

# [코랩 최적화] 코랩 웹브라우저 안에서 3D 렌더링이 깨지지 않도록 고정하는 설정입니다.
fig.show(renderer="colab")

# 결과물 HTML 저장
fig.write_html(os.path.join(OUTPUT, 'complex_3d_visualization.html'))
print(f"3D 시각화 완료: {OUTPUT}/complex_3d_visualization.html")


# --- 링크 예측 테스트 (Link Prediction) ---
print("\n--- 링크 예측 테스트 (Link Prediction) ---")
relations = list(result.training.relation_to_id.keys())
target_rel = [r for r in relations if 'diplom' in r.lower()][0] 
print(f"예측에 사용할 실제 관계 이름: {target_rel}")

df_tail = pykeen.predict.predict_target(
    model=result.model,
    head="brazil",
    relation=target_rel, 
    triples_factory=result.training,
).df

print(df_tail.head(10))