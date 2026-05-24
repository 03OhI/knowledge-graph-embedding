"""
ComplEx evaluation & visualization
학습이 끝난 후 저장된 모델을 로드해서 평가 + 시각화 수행.

Usage:
    python evaluate_visualize.py
    python evaluate_visualize.py --output ./complex_result --sample_size 1500
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import plotly.express as px
from sklearn.manifold import TSNE

import pykeen.predict
from pykeen.triples import TriplesFactory


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, 'complex_result')


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='학습 결과가 저장된 디렉토리')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='3D 시각화 샘플 개체 수')
    parser.add_argument('--target_head', type=str, default='brazil',
                        help='링크예측 head (없으면 첫 번째 엔티티로 자동 대체)')
    parser.add_argument('--target_relation', type=str, default='diplom',
                        help='링크예측 관계 키워드 (포함하는 첫 관계 사용)')
    args = parser.parse_args()

    OUTPUT = args.output
    if not os.path.isdir(OUTPUT):
        raise FileNotFoundError(f"결과 디렉토리를 찾을 수 없습니다: {OUTPUT}")
    print(f"결과 디렉토리: {OUTPUT}")
    print(f"포함 파일: {os.listdir(OUTPUT)}")

    # ── 저장된 모델 & 학습 팩토리 로드 ──────────────────────
    model_path = os.path.join(OUTPUT, 'trained_model.pkl')
    triples_path = os.path.join(OUTPUT, 'training_triples')
    results_path = os.path.join(OUTPUT, 'results.json')

    print("\n모델 로딩 중...")
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    training_factory = TriplesFactory.from_path_binary(triples_path)
    model.eval()

    # ── 평가 결과 출력 ─────────────────────────────────────
    print("\n--- 모델 성능 평가 ---")
    with open(results_path, 'r') as f:
        results_data = json.load(f)
    metrics = results_data.get('metrics', results_data)
    flat = flatten_dict(metrics)

    mrr = flat.get('both.realistic.inverse_harmonic_mean_rank')
    hits_at_10 = (flat.get('both.realistic.hits_at_10')
                  or flat.get('both.avg.hits_at_10'))

    print(f"MRR: {mrr:.4f}" if mrr is not None else "MRR 키를 찾을 수 없음.")
    print(f"Hits@10: {hits_at_10:.4f}" if hits_at_10 is not None else "Hits@10 키를 찾을 수 없음.")

    # ── 임베딩 추출 (ComplEx도 복소수) ────────────────────
    print("\n시각화 준비. 임베딩 추출 및 차원 축소 실행")

    entity_to_id = training_factory.entity_to_id
    entity_names = list(entity_to_id.keys())
    entity_ids = torch.tensor(list(entity_to_id.values()), device=model.device)
    complex_embeddings = model.entity_representations[0](entity_ids).detach().cpu().numpy()

    # 복소수 [a + bi] → [a, b] 실수 벡터
    real_part = np.real(complex_embeddings)
    imag_part = np.imag(complex_embeddings)
    entity_embeddings = np.concatenate([real_part, imag_part], axis=-1)

    # ── 3D t-SNE 시각화 (샘플링) ──────────────────────────
    print("3차원 차원 축소 중...")
    SAMPLE_SIZE = min(args.sample_size, len(entity_names))
    np.random.seed(42)
    sample_indices = np.random.choice(len(entity_names), SAMPLE_SIZE, replace=False)
    sampled_embeddings = entity_embeddings[sample_indices]
    sampled_entity_names = [entity_names[i] for i in sample_indices]

    tsne_3d = TSNE(
        n_components=3,
        random_state=42,
        perplexity=min(30, len(sampled_entity_names) - 1),
        max_iter=1000,
    )
    X_3d = tsne_3d.fit_transform(sampled_embeddings)

    df_3d = pd.DataFrame(X_3d, columns=['x', 'y', 'z'])
    df_3d['entity'] = sampled_entity_names

    fig = px.scatter_3d(
        df_3d, x='x', y='y', z='z',
        text='entity', color='x',
        title='ComplEx Entity Embeddings (3D t-SNE) - Sampled',
        labels={'x': 'TSNE-1', 'y': 'TSNE-2', 'z': 'TSNE-3'},
    )
    fig.update_traces(marker=dict(size=5), textposition='top center')
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    html_path = os.path.join(OUTPUT, 'complex_3d_visualization.html')
    fig.write_html(html_path)
    print(f"3D 시각화 완료: {html_path}")

    # ── 링크 예측 ─────────────────────────────────────────
    print("\n--- 링크 예측 테스트 (Link Prediction) ---")
    relations = list(training_factory.relation_to_id.keys())

    matching_rels = [r for r in relations if args.target_relation.lower() in r.lower()]
    target_rel = matching_rels[0] if matching_rels else relations[0]
    print(f"사용할 관계: {target_rel}")

    target_head = args.target_head if args.target_head in entity_names else entity_names[0]
    print(f"사용할 Head 엔티티: {target_head}")

    df_tail = pykeen.predict.predict_target(
        model=model,
        head=target_head,
        relation=target_rel,
        triples_factory=training_factory,
    ).df

    print(f"\n[{target_head}] ---({target_rel})---> [?] Tail 예측 결과 (Top 10):")
    print(df_tail.head(10))

    csv_path = os.path.join(OUTPUT, 'link_prediction_top10.csv')
    df_tail.head(10).to_csv(csv_path, index=False)
    print(f"\n예측 결과 저장: {csv_path}")


if __name__ == '__main__':
    main()