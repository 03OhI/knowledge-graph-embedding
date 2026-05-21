!pip install pykeen
!pip install matplotlib
!pip install plotly
!pip install seaborn
!pip install torch
from google.colab import drive
drive.mount('/content/drive')

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
from pykeen.losses import NSSALoss
from pykeen.stoppers import EarlyStopper
OUTPUT = '/content/drive/MyDrive/rotate_result'
os.makedirs(OUTPUT, exist_ok=True)
print("저장 경로:", OUTPUT)
print("기존 파일:", os.listdir(OUTPUT))   # 체크포인트 있는지 확인

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"장치: {device}")
print("RotatE 학습 시작...")

result = pipeline(
    dataset='FB15k237',
    model='RotatE',
    model_kwargs=dict(embedding_dim=1000),
    loss=NSSALoss,
    loss_kwargs=dict(margin=9.0, adversarial_temperature=1.0),
    negative_sampler_kwargs=dict(num_negs_per_pos=256),
    training_kwargs=dict(
        num_epochs=1000,
        batch_size=512,
        sub_batch_size=256,
        use_tqdm=True,
        checkpoint_name='rotate_fb15k237.pt',
        checkpoint_directory=OUTPUT,
        checkpoint_frequency=5,        # 30 → 5 (자주 저장)
    ),
    optimizer_kwargs=dict(lr=2e-5),
    stopper='early',
    stopper_kwargs=dict(
        frequency=25, patience=5,
        relative_delta=0.0001, metric='mrr',
    ),
    evaluator_kwargs=dict(filtered=True),
    evaluation_kwargs=dict(batch_size=8),
    device=device,
    random_seed=42,
)

result.save_to_directory(OUTPUT)
print(f"모델과 결과를 '{OUTPUT}'에 저장.")

# 1) 지금 import된 것들이 살아있는지
try:
    print("pipeline 존재:", pipeline)
    print("torch 존재:", torch.__version__)
except NameError as e:
    print("리셋됨:", e)

# 2) 현재 GPU 메모리 상태
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"메모리 사용: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")

# 3) 지금 셀이 실행 중인지, 아니면 멈춰 있는지 확인
# (학습 셀에 ■ 표시면 실행 중, ▶면 멈춤)

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