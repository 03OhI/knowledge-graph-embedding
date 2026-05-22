# knowledge-graph-embedding
지식그래프 임베딩 해보기

# 참고!!
작업을 하면서 새로운 라이브러리 설치가 필요하기 때문에.. 실행이 안된다면 콘다 환경 생성을 다시 해주세요. 제가 버전 올릴 때마다 환경 업데이트 해두겠습니다.
## 환경 설치법
모두 conda는 사용해보셨을거에요! 딥러닝이나 데이터 분석같은 경우 venv를 사용하는 것보단 conda를 사용하는게 더 안정적이에요.

아시다시피 pykeen 라이브러리 설치를 해야하는데, pykeen도 일종의 딥러닝이여서 sklearn도 설치를 했습니다.(파이썬 3.12.x, numpy 2.0이하 필수)

근데, 어차피 제가 환경을 추출해뒀으니 여러분은 아래 프로젝트 파일에서 명령어만 실행시키면 될거에요. (conda는 미리 설치해주세요)
```
#콘다 환경 생성. KGE는 KnowledgeGraphEmbeddig 약자입니다.
conda KGE create -f environment.yml
#콘다 환경 실행
conda activate KGE
```
실행은.. 간단하게
```
python main.py
```
---
# ComplEx1.0.0!!
마지막 CompEx1.0.0 모델 구현까지 완료했습니다. 전에 만들었던 RotatE1.0.0 버전과 모델명만 제외하면 완전히 동일하게 구현이 가능했습니다.
```
result = pipeline(
    dataset='nations',
    model='complex', <-- 모델명만 변경
    model_kwargs=dict(
        embedding_dim=50,
    ),
    training_kwargs=dict(
        num_epochs=100,
        batch_size=64,
        use_tqdm=True,
    ),
    optimizer_kwargs=dict(lr=1e-2),
    device='cpu'
)
```

실행결과, ComplEx는 TransE와 RotatE의 사이 정도의 성능지표 값이 나왔습니다. MRR은 0.54, Hits@10은 0.97로 나왔습니다. ComplEx는 비대칭 관계에 가능하기 때문에 데이터셋이 달라진다면 가장 좋게 나올 수도 있겠다 생각이 듭니다.

---
# RotatE1.0.0!!
RotatE 모델 또한 구현을 완료했습니다. 라이브러리에 내장된 모델을 사용하기 때문에 TransE 구현과 큰 차이는 없었으나, TransE는 거리, 즉 벡터를 최소화하는 방식이였기에 norm 옵션이 필요했지만 RotatE는 복소수 공간에서 회전시키기 때문에 norm 옵션이 필요하지 않습니다. TransE1.1.0 버전에서 수정하였기에 나머지는 같은 기능을 제공합니다.
```
result = pipeline(
    dataset='nations',
    model='RotatE', <-- 모델명만 변경
    model_kwargs=dict(
        embedding_dim=50,
        # scoring_fct_norm=1,  <-- 이 옵션이 필요하지 않음.
    ),
    training_kwargs=dict(
        num_epochs=100,
        batch_size=64,
        use_tqdm=True,
    ),
    optimizer_kwargs=dict(lr=1e-2),
    device='cpu'
)
```

실행결과, 성능 평가 지표인 MRR과 Hits@10의 값이 기존 TrnasE와 차이가 있었습니다. TrnasE의 경우 MRR은 약 0.45, Hits@10은 약 0.97 이었지만 RotatE는 MRR이 0.60, Hits@10이 0.98로 올랐습니다. MRR에서 큰 차이가 나타났습니다. 분명 시각화환 3D 산점도도 차이가 있을텐데.. 막눈인 제가 봤을 땐 큰 차이를 못느끼겠더라구요...ㅎ 데이터셋이 단순한 이유이기도 할 것 같습니다.

---
# TransE1.1.0!!
2차원 시각화에서 3차원 시각화로 변경하였습니다. 또한, predict_target() 함수를 사용하여 brazil이 외교관을 추방할 가능성이 높은 국가들을 찾아보도록 했습니다. 

결과는 직접 실행해보기~
```
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
```

---
# TransE1.0.0!!
아기다리 고기다리 TransE1.0.0을 샘플데이터에 학습 시키고 지식 그래프를 차원 축소하여 시각화까지 마쳤습니다.
### 모델정의
pykeen에서 제공하는 pipeline 함수는 자동으로 파라미터값을 맞춰주기 때문에, 간단하게 모델을 정의할 수 있었습니다.
```
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
```

애를 먹었던건 성능 평가였는데, MRR과 Hits@10을 측정할 때, result에서 나오는 키값이 pykeen의 버전에 따라 계속 바뀌어서.. 한참을 찾았습니다.

또한, 현재는 시각화를 matplotlib와 seaborn을 사용했는데, 2차원 산점도론 확 와닿지 않는 결과인것 같습니다.

### 객체 임베딩 시각화
<img width="1312" height="1176" alt="image" src="https://github.com/user-attachments/assets/c084963b-feec-4b28-bfa8-d9937d44fa35" />

### 관계 임베딩 시각화
<img width="1112" height="976" alt="image" src="https://github.com/user-attachments/assets/ce29ba21-5d68-41bc-ab66-cf92c9ba2d47" />

---

# 결과
## nations 데이터셋 학습 후 정확도 측정
|모델/지표|MRR|Hits@10|
|------|---|--------|
|TransE|0.45|0.97|
|RotatE|0.60|0.98|
|ComplEx|0.54|0.97|

위 표를 보듯 nations 데이터셋에서는 RotatE가 가장 관계를 잘 나타냈다고 볼 수 있다.

## 3D 시각화


