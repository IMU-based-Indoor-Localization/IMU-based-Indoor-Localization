import torch
import torch.nn as nn

class MTL_CustomLoss(nn.Module):
    def __init__(self, lambda_pos=10.0, lambda_unc=1.0, lambda_cls=1.0):
        super(MTL_CustomLoss, self).__init__()
        self.lambda_pos = lambda_pos
        self.lambda_unc = lambda_unc
        self.lambda_cls = lambda_cls
        
        self.mse_loss = nn.MSELoss()
        # 라벨이 -1인 데이터는 자세 분류 학습에서 제외
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, pred_mu, pred_log_var, target_reg, pred_cls, target_cls):
        
        loss_mse = self.mse_loss(pred_mu, target_reg)
        
        precision = torch.exp(-pred_log_var)
        loss_nll_elements = 0.5 * pred_log_var + 0.5 * precision * (target_reg - pred_mu)**2
        loss_nll = torch.mean(loss_nll_elements)
        
        loss_cls = self.ce_loss(pred_cls, target_cls)
        
        total_loss = (self.lambda_pos * loss_mse) + \
                     (self.lambda_unc * loss_nll) + \
                     (self.lambda_cls * loss_cls)
                     
        return total_loss, loss_mse, loss_nll, loss_cls