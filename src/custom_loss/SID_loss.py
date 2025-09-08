### From ChemProp ##
# Modified #
import torch
import torch.nn as nn

def sid_loss(
    model_spectra: torch.Tensor,   # (B, L)
    target_spectra: torch.Tensor,  # (B, L)
    mask: torch.Tensor,            # (B, L) bool or 0/1
    eps: float = 1e-8,
    reduction: str = "mean_valid", # "mean_valid" | "mean" | "sum" | "none"
) -> torch.Tensor:
    """
    SID with both p,q normalized *inside mask* and masked-out terms removed.
    - p = model / sum(model*mask)
    - q = target / sum(target*mask)
    - sid_i = sum_j [ p_j (log p_j - log q_j) + q_j (log q_j - log p_j) ] over masked j
    - reduction:
        "mean_valid": (per-sample sum / valid_count) -> batch mean   [추천: [GRAD] SID와 일치]
        "mean":       전체(B*L) 평균(마스크 밖은 0)                   [chemprop의 .mean()에 가까움]
        "sum":        합만 반환
        "none":       (B, L) 위치별 항 반환
    """
    device = model_spectra.device
    # mask -> bool
    mask = mask.bool().to(device)

    # 마스크 안쪽만 사용, log 안전을 위해 eps 바닥
    p_raw = (model_spectra.clamp_min(eps)) * mask
    q_raw = (target_spectra.clamp_min(eps)) * mask

    # 마스크 합으로 각 분포 정규화
    p_sum = p_raw.sum(dim=1, keepdim=True).clamp_min(eps)
    q_sum = q_raw.sum(dim=1, keepdim=True).clamp_min(eps)
    p = p_raw / p_sum
    q = q_raw / q_sum

    # SID 원식: p*log(p/q) + q*log(q/p)
    logp = p.clamp_min(eps).log()
    logq = q.clamp_min(eps).log()
    sid_elem = p * (logp - logq) + q * (logq - logp)  # (B, L)

    if reduction == "none":
        return sid_elem * mask  # 위치별 항

    if reduction == "sum":
        return (sid_elem * mask).sum()

    if reduction == "mean":
        # 전체 포인트(B*L) 평균(마스크 바깥은 0)
        B, L = sid_elem.shape
        return (sid_elem * mask).sum() / (B * L)

    if reduction == "mean_valid":
        # 각 샘플의 유효 길이로 나눈 뒤 배치 평균 → [GRAD] SID와 동일 스케일
        valid_counts = mask.sum(dim=1).clamp_min(1)
        per_sample = (sid_elem * mask).sum(dim=1) / valid_counts
        return per_sample.mean()

    raise ValueError(f"Unknown reduction: {reduction}")

## old_version ##
#def sid_loss(
#    model_spectra: torch.tensor,
#    target_spectra: torch.tensor,
#    mask: torch.tensor,
#    threshold: float = None,
#) -> torch.tensor:
#    """
#    Loss function for use with spectra data type.
#
#    :param model_spectra: The predicted spectra output from a model with shape (batch_size,spectrum_length).
#    :param target_spectra: The target spectra with shape (batch_size,spectrum_length). Values must be normalized so that each spectrum sums to 1.
#    :param mask: Tensor with boolean indications of where the spectrum output should not be excluded with shape (batch_size,spectrum_length).
#    :param threshold: Loss function requires that values are positive and nonzero. Values below the threshold will be replaced with the threshold value.
#    :return: A tensor containing loss values for the batch with shape (batch_size,spectrum_length).
#    """
#    # Move new tensors to torch device
#    torch_device = model_spectra.device
#
#    # Normalize the model spectra before comparison
#    zero_sub = torch.zeros_like(model_spectra, device=torch_device)
#    one_sub = torch.ones_like(model_spectra, device=torch_device)
#    if threshold is not None:
#        threshold_sub = torch.full(model_spectra.shape, threshold, device=torch_device)
#        model_spectra = torch.where(model_spectra < threshold, threshold_sub, model_spectra)
#    model_spectra = torch.where(mask, model_spectra, zero_sub)
#    sum_model_spectra = torch.sum(model_spectra, axis=1, keepdim=True)
#    model_spectra = torch.div(model_spectra, sum_model_spectra)
#
#    # Calculate loss value
#    target_spectra = torch.where(mask, target_spectra, one_sub)
#    model_spectra = torch.where(mask, model_spectra, one_sub)  # losses in excluded regions will be zero because log(1/1) = 0.
#    loss = torch.mul(torch.log(torch.div(model_spectra, target_spectra)), model_spectra) + torch.mul(
#        torch.log(torch.div(target_spectra, model_spectra)), target_spectra
#    )
#
#    return loss