import os
import torch
import pandas as pd
import numpy as np
from pykeen.triples import TriplesFactory
from tqdm import tqdm

# 1. 멀티코어 설정
torch.set_num_threads(8) 

OUTPUT = 'rotate_result' 
model_path = os.path.join(OUTPUT, 'trained_model.pkl')
triples_path = os.path.join(OUTPUT, 'training_triples')

def main():
    device = torch.device("cpu")
    print(f"🚀 초경량 모드로 모델 로딩 중...")
    
    model = torch.load(model_path, map_location=device)
    training_factory = TriplesFactory.from_path_binary(triples_path)
    model.eval()

    entity_id_to_label = {v: k for k, v in training_factory.entity_to_id.items()}
    relation_id_to_label = {v: k for k, v in training_factory.relation_to_id.items()}
    
    # ── [방법 2 적용] 엔티티 샘플링 ────────────────────────────────────
    # 전체 1.4만 개 중 1,000개만 무작위로 뽑습니다. (속도 14배 향상)
    num_entities = model.num_entities
    sample_size = 1000 
    torch.manual_seed(42) # 결과 재현을 위해 시드 고정
    sampled_indices = torch.randperm(num_entities)[:sample_size]
    # ──────────────────────────────────────────────────────────────────

    # ── [방법 1 적용] 중요 관계 필터링 ──────────────────────────────────
    # FB15k-237에서 의미 있는 관계 키워드 몇 개만 지정합니다.
    # 만약 모든 관계를 보고 싶다면 이 리스트를 비우고 아래 if문을 수정하세요.
    target_keywords = ['location', 'nationality', 'contains', 'place', 'genre']
    all_relation_ids = torch.arange(model.num_relations, device=device)
    
    selected_relations = []
    for r_id in all_relation_ids:
        r_label = relation_id_to_label[r_id.item()]
        if any(kw in r_label.lower() for kw in target_keywords):
            selected_relations.append(r_id)
    
    if not selected_relations: # 키워드에 맞는 게 없으면 첫 10개라도 뽑음
        selected_relations = all_relation_ids[:10]
    # ──────────────────────────────────────────────────────────────────

    print(f"✅ 설정 완료: 엔티티 {sample_size}개, 관계 {len(selected_relations)}개 추론")
    
    results = []
    head_batch_size = 100 # 샘플링을 했으므로 100이면 충분히 쾌적합니다.

    with torch.no_grad():
        for r_id in tqdm(selected_relations, desc="[1/2] 필터링된 관계 진행"):
            r_label = relation_id_to_label[r_id.item()]
            
            # 샘플링된 엔티티들에 대해서만 루프
            for start_idx in range(0, sample_size, head_batch_size):
                end_idx = min(start_idx + head_batch_size, sample_size)
                h_ids_batch = sampled_indices[start_idx:end_idx].to(device)
                
                hr_batch = torch.stack([
                    h_ids_batch, 
                    r_id.expand(len(h_ids_batch))
                ], dim=1)

                scores = model.score_t(hr_batch)
                best_tail_indices = torch.argmax(scores, dim=1).cpu().numpy()
                best_scores = torch.max(scores, dim=1).values.cpu().numpy()

                for i, (t_id, score) in enumerate(zip(best_tail_indices, best_scores)):
                    h_id = h_ids_batch[i].item()
                    results.append({
                        'head': entity_id_to_label[h_id],
                        'relation': r_label,
                        'tail': entity_id_to_label[t_id],
                        'score': float(score)
                    })

    print("\n[2/2] 샘플링 데이터 저장 중...")
    final_df = pd.DataFrame(results)
    save_name = 'sampled_top1_results.csv'
    final_df.to_csv(os.path.join(OUTPUT, save_name), index=False)
    print(f"✨ 완료! '{save_name}' 파일을 확인하세요.")

if __name__ == '__main__':
    main()