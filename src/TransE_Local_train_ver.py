"""
TransE training on FB15k-237 (local GPU version)
Usage:
    python train_transe.py
    python train_transe.py --output ./transe_result --epochs 500
"""

import os
import argparse
import torch

from pykeen.pipeline import pipeline
from pykeen.losses import NSSALoss

# 스크립트 파일이 있는 디렉토리 기준 (어디서 실행하든 동일)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, 'transe_result')


def main():
    parser = argparse.ArgumentParser(description='TransE training on FB15k-237')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='결과/체크포인트 저장 디렉토리 (기본: 스크립트 옆 transe_result/)')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--sub_batch_size', type=int, default=512)
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='TransE는 200~400 정도가 일반적 (ComplEx보다 작게)')
    parser.add_argument('--scoring_fct_norm', type=int, default=1,
                        help='거리 함수 norm (FB15k-237에서는 L1=1이 보통 가장 좋음)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='TransE는 ComplEx보다 큰 lr이 잘 먹음')
    parser.add_argument('--num_negs', type=int, default=64)
    parser.add_argument('--margin', type=float, default=9.0)
    parser.add_argument('--adv_temp', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ── 출력 디렉토리 준비 (없으면 자동 생성) ──────────────────
    OUTPUT = args.output
    os.makedirs(OUTPUT, exist_ok=True)
    print(f"저장 경로: {OUTPUT}")
    print(f"기존 파일: {os.listdir(OUTPUT)}")  # 체크포인트 있으면 자동 이어서 학습됨

    # ── 디바이스 설정 ─────────────────────────────────────────
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        print(f"장치: {device}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_mem:.2f}GB")
    else:
        device = torch.device("cpu")
        print("⚠️  CUDA 사용 불가 — CPU로 실행됩니다 (매우 느림)")

    # ── 학습 ──────────────────────────────────────────────────
    # TransE 핵심 차이점:
    #  • 임베딩 차원: ComplEx(1000) → TransE(256) 정도가 적절
    #  • scoring_fct_norm: L1(=1) 또는 L2(=2). FB15k-237은 L1이 보통 최적
    #  • 정규화: TransE는 entity embedding을 unit norm으로 제약하기 때문에
    #            LpRegularizer 없이도 안정적으로 학습됨 → regularizer 제거
    #  • learning rate: ComplEx(2e-5) → TransE는 1e-3 수준으로 더 크게
    print("TransE 학습 시작...")
    result = pipeline(
        dataset='FB15k237',
        model='TransE',
        model_kwargs=dict(
            embedding_dim=args.embedding_dim,
            scoring_fct_norm=args.scoring_fct_norm,
        ),
        loss=NSSALoss,
        loss_kwargs=dict(margin=args.margin, adversarial_temperature=args.adv_temp),
        negative_sampler_kwargs=dict(num_negs_per_pos=args.num_negs),
        training_kwargs=dict(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            sub_batch_size=args.sub_batch_size,
            use_tqdm=True,
            checkpoint_name='transe_fb15k237.pt',
            checkpoint_directory=OUTPUT,
            checkpoint_frequency=5,
        ),
        optimizer_kwargs=dict(lr=args.lr),
        stopper='early',
        stopper_kwargs=dict(
            frequency=25, patience=5,
            relative_delta=0.0001, metric='mrr',
        ),
        evaluator_kwargs=dict(filtered=True),
        evaluation_kwargs=dict(batch_size=16),
        device=device,
        random_seed=args.seed,
    )

    # ── 결과 저장 ─────────────────────────────────────────────
    result.save_to_directory(OUTPUT)
    print(f"\n✅ 모델과 결과를 '{OUTPUT}'에 저장 완료")

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"최종 VRAM 사용: {used:.2f}GB / {total:.2f}GB")


if __name__ == '__main__':
    main()
