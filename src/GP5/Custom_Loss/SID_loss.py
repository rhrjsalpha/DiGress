import torch
import torch.nn as nn


class SIDLoss(nn.Module):
    def __init__(self):
        super(SIDLoss, self).__init__()

    def forward(self, y_pred, y_target):
        # 아주 작은 값을 추가하여 로그 연산의 안정성을 보장
        epsilon = 1e-8
        epsilon_clmap = 1e-4
        print("y_pred_sid",y_pred)
        print("y_target_sid",y_target)

        # 입력이 3차원인 경우 텐서 크기 변환: [batch_size, seq_len, feature_dim]
        if y_pred.dim() == 3 :
            y_pred = y_pred.view(-1, y_pred.size(-1))  # [batch_size * seq_len, feature_dim]
        if y_target.dim() == 3:
            y_target = y_target.view(-1, y_target.size(-1))

        print("y_pred_dim",y_pred)
        print("y_target_dim",y_target)

        # 정규화 (각 벡터의 합이 1이 되도록)
        n = y_pred.size(0)
        y_pred_clamped = torch.clamp(y_pred, min=epsilon_clmap)
        print("y_pred_clamped",y_pred_clamped)
        y_pred_sum = y_pred_clamped.sum(dim=1, keepdim=False)
        y_pred_sum = y_pred_sum.sum(dim=0, keepdim=False)
        print("y_pred_sum",y_pred_sum)
        y_pred_sum = y_pred_sum.unsqueeze(0).repeat(n, 1)
        y_pred = y_pred / (y_pred_sum+epsilon)


        print("y_pred_norm",y_pred)

        y_target_sum = y_target.sum(dim=1, keepdim=False)
        y_target_sum = y_target_sum.sum(dim=0, keepdim=False)
        y_target_sum = y_target_sum.unsqueeze(0).repeat(n, 1)
        y_target = y_target / (y_target_sum + epsilon)
        print("y_target_norm",y_target)

        # SID 계산
        divergence = (
            (y_pred * torch.log((y_pred + epsilon) / (y_target + epsilon))).sum(dim=1) +
            (y_target * torch.log((y_target + epsilon) / (y_pred + epsilon))).sum(dim=1)
        )
        #print("divergence", divergence)
        return divergence.mean()

if __name__ == "__main__":
    # 예측값과 실제값 (Batch 크기: 2, 길이: 5)
    y_pred = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.0], [0.2, 0.1, 0.0, 0.3, 0.4]])
    y_target = torch.tensor([[0.1, 0.2, 0.2, 0.5, 0.0], [0.2, 0.1, 0.1, 0.3, 0.3]])

    # SIDLoss 초기화 및 계산
    criterion = SIDLoss()
    loss = criterion(y_pred, y_target)
    print(f"SID Loss: {loss.item()}")