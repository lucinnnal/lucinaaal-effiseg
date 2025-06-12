
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers.trainer import Trainer
from src.utils.loss import CrossEntropyLoss2d
from typing import Dict, List, Tuple, Optional, Any, Union
from src.utils.loss import DiceFocalLoss
from src.utils.loss import DiceFocalLoss_updated
from src.utils.counts.count import PixelCounter 

class BaseTrainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.loss_fn = CrossEntropyLoss2d()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    def compute_loss(self, model, inputs, num_items_in_batch = None, return_outputs=False):
        image = inputs['pixel_values'].to(self.device)
        label = inputs['labels'].to(self.device)
        # Forward pass
        output = model(image) # output is a dictionary (feature map :  ,logits:torch.Size([2, 20, 128, 256]), patch embeddings: torch.Size([2, 32768, 32], torch.Size([2, 8192, 64])...))
        # Compute loss
        loss = self.loss_fn(output, label)

        if return_outputs:
            return loss, output, label
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = ["patch_embeddings", "feature_map"],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: # Optional은 Tensor 또는 None이 올 수 있다는 뜻 
        
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model,inputs,return_outputs = True)
        
        return (eval_loss,pred,label)

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=1.0, alpha=0.9, **kwds):
        super().__init__(**kwds)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Teacher model load for distillation
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval() 
        
        # Distillation parameters
        self.temperature = temperature
        self.alpha = alpha  # distillation loss weight
        
        # Loss function (기존 BaseTrainer와 동일)
        self.loss_fn = CrossEntropyLoss2d()
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        compute loss including label loss + distillation losses
        """
        image = inputs['pixel_values'].to(self.device)
        label = inputs['labels'].to(self.device)
        
        # student forward pass
        student_output = model(image)
        
        # teacher forward pass => eval mode
        with torch.no_grad():
            teacher_output = self.teacher_model(image)
        
        # Target loss 
        hard_loss = self.loss_fn(student_output, label)
        
        # Pixelwise Distillation Loss
        soft_loss = self._distillation_loss(student_output, teacher_output)
        
        # 3. Combined loss
        total_loss = hard_loss + self.alpha * soft_loss
        
        if return_outputs:
            return total_loss, student_output, label
        return total_loss
    
    def _distillation_loss(self, student_logits, teacher_logits):
        """
        Segmentation Knowledge Distillation Loss
        Ver1. total loss = label loss + pixel level KL divergence distillation loss
        """
        student_logits = student_logits['logits']
        teacher_logits = teacher_logits['logits']

        # Logits shape: [batch_size, num_classes, height, width]
        batch_size, num_classes, height, width = student_logits.shape
        
        # Flatten spatial dimensions for easier computation
        student_flat = student_logits.view(batch_size, num_classes, -1)  # [B, C, H*W]
        teacher_flat = teacher_logits.view(batch_size, num_classes, -1)  # [B, C, H*W]
        
        # Temperature scaling and softmax
        student_soft = F.softmax(student_flat / self.temperature, dim=1)  # [B, C, H*W]
        teacher_soft = F.softmax(teacher_flat / self.temperature, dim=1)  # [B, C, H*W]
        
        # KL Divergence Loss
        kl_loss = F.kl_div(F.log_softmax(student_flat / self.temperature, dim=1), teacher_soft, reduction='batchmean')
        kl_loss = kl_loss * (self.temperature ** 2)
        kl_loss = kl_loss / (height * width)
        
        # Temperature^2 scaling
        return kl_loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = ["patch_embeddings", "feature_map"],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Evaluation step with distillation
        """
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model, inputs, return_outputs=True)
        
        return (eval_loss, pred, label)

# Hard loss + Soft loss + Patch Embedding loss 
class PEdistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=1.0, alpha=1.0, beta = 1.0, **kwds):
        super().__init__(**kwds)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Teacher model load for distillation
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval() 
        
        # Distillation parameters
        self.temperature = temperature
        self.alpha = alpha  # distillation loss weight (KL)
        self.beta = beta # PE Loss weight        
        
        # Loss function (기존 BaseTrainer와 동일)
        self.loss_fn = CrossEntropyLoss2d()
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        compute loss including label loss + distillation losses
        """
        image = inputs['pixel_values'].to(self.device)
        label = inputs['labels'].to(self.device)
        
        # student forward pass
        student_output = model(image)
        
        # teacher forward pass => eval mode
        with torch.no_grad():
            teacher_output = self.teacher_model(image)
        
        # Target loss 
        hard_loss = self.loss_fn(student_output, label)
        
        # Pixelwise Distillation Loss
        soft_loss = self._distillation_loss(student_output, teacher_output)

        # PE Loss
        pe_loss = self._patchembedding_loss(student_output, teacher_output)
        
        # 3. Combined loss
        total_loss = hard_loss + self.alpha * soft_loss + self.beta * pe_loss
        
        if return_outputs:
            return total_loss, student_output, label
        return total_loss
    
    def _distillation_loss(self, student_logits, teacher_logits):
        """
        Segmentation Knowledge Distillation Loss
        Ver1. total loss = label loss + pixel level KL divergence distillation loss
        """
        # Logits
        student_logits = student_logits['logits']
        teacher_logits = teacher_logits['logits']

        # Logits shape: [batch_size, num_classes, height, width]
        batch_size, num_classes, height, width = student_logits.shape
        
        # Flatten spatial dimensions for easier computation
        student_flat = student_logits.view(batch_size, num_classes, -1)  # [B, C, H*W]
        teacher_flat = teacher_logits.view(batch_size, num_classes, -1)  # [B, C, H*W]
        
        # Temperature scaling and softmax
        student_soft = F.softmax(student_flat / self.temperature, dim=1)  # [B, C, H*W]
        teacher_soft = F.softmax(teacher_flat / self.temperature, dim=1)  # [B, C, H*W]
        
        # KL Divergence Loss
        kl_loss = F.kl_div(F.log_softmax(student_flat / self.temperature, dim=1), teacher_soft, reduction='batchmean')
        kl_loss = kl_loss * (self.temperature ** 2)
        kl_loss = kl_loss / (height * width)
        
        return kl_loss
    
    def _patchembedding_loss(self, student_outputs, teacher_outputs, embedding_weights = [0.1,0.1,0.5,1]):
        # Loss
        emb_loss = 0
        loss_embedding = nn.MSELoss()

        # Patch Embeddings
        student_emb = student_outputs['patch_embeddings']
        teacher_emb = teacher_outputs['patch_embeddings']

        for i in range(len(student_emb)): emb_loss += embedding_weights[i] * loss_embedding(student_emb[i], teacher_emb[i])

        return emb_loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = ["patch_embeddings", "feature_map"],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Evaluation step with distillation
        """
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model, inputs, return_outputs=True)
        
        return (eval_loss, pred, label)

# Hard loss + Soft loss + Patch Embedding loss + Feature Map Distillation loss(MSELoss)
class FeatureDistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=1.0, alpha=1.0, beta=1.0, gamma=1.0, 
                 teacher_channels=[64, 128, 320, 512], student_channels=[32, 64, 160, 256], **kwds): # -> default channel set to b2, b0 channels
        super().__init__(**kwds)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Teacher model load for distillation
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval() 
        
        # Distillation parameters
        self.temperature = temperature
        self.alpha = alpha  # distillation loss weight (KL)
        self.beta = beta    # PE Loss weight
        self.gamma = gamma  # Feature map distillation weight
        
        # Loss function -> Target Loss
        self.loss_fn = CrossEntropyLoss2d()
        
        # Channel Informations of Student and Teacher
        self.teacher_channels = teacher_channels
        self.student_channels = student_channels
        
        # Channel projection layers -> 1*1 Convolution
        self.projection_layers = nn.ModuleList()
        for t_ch, s_ch in zip(teacher_channels, student_channels):
            if t_ch != s_ch:
                proj = nn.Sequential(
                    nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(t_ch),
                    nn.ReLU()
                )
            else:
                proj = nn.Identity()
            self.projection_layers.append(proj)
        
        # Projection layers for each stage to device
        self.projection_layers.to(self.device)
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        compute loss including label loss + distillation losses
        """
        image = inputs['pixel_values'].to(self.device)
        label = inputs['labels'].to(self.device)
        
        # student forward pass
        student_output = model(image)
        
        # teacher forward pass => eval mode
        with torch.no_grad():
            teacher_output = self.teacher_model(image)
        
        # Target loss 
        hard_loss = self.loss_fn(student_output, label)
        
        # Pixelwise Distillation Loss
        soft_loss = self._distillation_loss(student_output, teacher_output)

        # PE Loss
        pe_loss = self._patchembedding_loss(student_output, teacher_output)
        
        # Feature Map Distillation Loss
        feature_loss = self._feature_distillation_loss(student_output, teacher_output)
        
        # Combined loss
        total_loss = hard_loss + self.alpha * soft_loss + self.beta * pe_loss + self.gamma * feature_loss
        
        if return_outputs:
            return total_loss, student_output, label
        return total_loss
    
    def _distillation_loss(self, student_logits, teacher_logits):
        """
        Segmentation Knowledge Distillation Loss
        Ver1. total loss = label loss + pixel level KL divergence distillation loss
        """
        # Logits
        student_logits = student_logits['logits']
        teacher_logits = teacher_logits['logits']

        # Logits shape: [batch_size, num_classes, height, width]
        batch_size, num_classes, height, width = student_logits.shape
        
        # Flatten spatial dimensions for easier computation
        student_flat = student_logits.view(batch_size, num_classes, -1)  # [B, C, H*W]
        teacher_flat = teacher_logits.view(batch_size, num_classes, -1)  # [B, C, H*W]
        
        # Temperature scaling and softmax
        student_soft = F.softmax(student_flat / self.temperature, dim=1)  # [B, C, H*W]
        teacher_soft = F.softmax(teacher_flat / self.temperature, dim=1)  # [B, C, H*W]
        
        # KL Divergence Loss
        kl_loss = F.kl_div(F.log_softmax(student_flat / self.temperature, dim=1), teacher_soft, reduction='batchmean')
        kl_loss = kl_loss * (self.temperature ** 2)
        kl_loss = kl_loss / (height * width)
        
        return kl_loss
    
    def _patchembedding_loss(self, student_outputs, teacher_outputs, embedding_weights=[0.1, 0.1, 0.5, 1]):
        # Loss
        emb_loss = 0
        loss_embedding = nn.MSELoss()

        # Patch Embeddings
        student_emb = student_outputs['patch_embeddings']
        teacher_emb = teacher_outputs['patch_embeddings']

        for i in range(len(student_emb)): 
            emb_loss += embedding_weights[i] * loss_embedding(student_emb[i], teacher_emb[i])

        return emb_loss
    
    def _feature_distillation_loss(self, student_outputs, teacher_outputs, feature_stage_weights=[1,1,1,1]):
        """
        Feature Map Distillation Loss
        각 stage의 feature map을 teacher로부터 student에게 distillation
        """
        # Feature maps 추출
        student_features = student_outputs['feature_map']  # List of 4 feature maps
        teacher_features = teacher_outputs['feature_map']  # List of 4 feature maps
        
        total_featuremap_loss = 0
        mse_loss = nn.MSELoss()
        
        for i, (student_feat, teacher_feat) in enumerate(zip(student_features, teacher_features)):
            # Channel projection student -> teacher channel 
            student_feat_proj = self.projection_layers[i](student_feat)
            
            # MSE loss between projected student features and teacher features
            stage_loss = mse_loss(student_feat_proj, teacher_feat.detach())
            
            # Weighted Stage loss
            weighted_stage_loss = feature_stage_weights[i] * stage_loss
            total_featuremap_loss += weighted_stage_loss
    
        return total_featuremap_loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = ["patch_embeddings", "feature_map"],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Evaluation step with distillation
        """
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model, inputs, return_outputs=True)
        
        return (eval_loss, pred, label)
    
# Hard loss + Patch Embedding loss(1.0) + Feature Map Distillation loss(Cosine Similarity loss) (0.5)
class CSKDTrainer(Trainer):
    def __init__(self, teacher_model, temperature=1.0, alpha=0.0, beta=1.0, gamma=0.5, 
                 teacher_channels=[64, 128, 320, 512], student_channels=[32, 64, 160, 256], **kwds): # -> default channel set to b2, b0 channels
        super().__init__(**kwds)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Teacher model load for distillation
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval() 
        
        # Distillation parameters
        self.temperature = temperature
        self.alpha = alpha  # distillation loss weight (KL)
        self.beta = beta    # PE Loss weight
        self.gamma = gamma  # Feature map distillation weight
        
        # Loss function -> Target Loss
        self.loss_fn = CrossEntropyLoss2d()
        
        # Channel Informations of Student and Teacher
        self.teacher_channels = teacher_channels
        self.student_channels = student_channels
        
        # Channel projection layers -> 1*1 Convolution
        self.projection_layers = nn.ModuleList()
        for t_ch, s_ch in zip(teacher_channels, student_channels):
            if t_ch != s_ch:
                proj = nn.Sequential(
                    nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(t_ch),
                    nn.ReLU()
                )
            else:
                proj = nn.Identity()
            self.projection_layers.append(proj)
        
        # Projection layers for each stage to device
        self.projection_layers.to(self.device)
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        compute loss including label loss + distillation losses
        """
        image = inputs['pixel_values'].to(self.device)
        label = inputs['labels'].to(self.device)
        
        # student forward pass
        student_output = model(image)
        
        # teacher forward pass => eval mode
        with torch.no_grad():
            teacher_output = self.teacher_model(image)
        
        # Target loss 
        hard_loss = self.loss_fn(student_output, label)
        
        # Pixelwise Distillation Loss
        soft_loss = self._distillation_loss(student_output, teacher_output)

        # PE Loss
        pe_loss = self._patchembedding_loss(student_output, teacher_output)
        
        # Feature Map Distillation Loss
        feature_loss = self._feature_distillation_loss(student_output, teacher_output)
        
        # Combined loss
        total_loss = hard_loss + self.alpha * soft_loss + self.beta * pe_loss + self.gamma * feature_loss
        
        if return_outputs:
            return total_loss, student_output, label
        return total_loss
    
    def _distillation_loss(self, student_logits, teacher_logits):
        """
        Segmentation Knowledge Distillation Loss
        Ver1. total loss = label loss + pixel level KL divergence distillation loss
        """
        # Logits
        student_logits = student_logits['logits']
        teacher_logits = teacher_logits['logits']

        # Logits shape: [batch_size, num_classes, height, width]
        batch_size, num_classes, height, width = student_logits.shape
        
        # Flatten spatial dimensions for easier computation
        student_flat = student_logits.view(batch_size, num_classes, -1)  # [B, C, H*W]
        teacher_flat = teacher_logits.view(batch_size, num_classes, -1)  # [B, C, H*W]
        
        # Temperature scaling and softmax
        student_soft = F.softmax(student_flat / self.temperature, dim=1)  # [B, C, H*W]
        teacher_soft = F.softmax(teacher_flat / self.temperature, dim=1)  # [B, C, H*W]
        
        # KL Divergence Loss
        kl_loss = F.kl_div(F.log_softmax(student_flat / self.temperature, dim=1), teacher_soft, reduction='batchmean')
        kl_loss = kl_loss * (self.temperature ** 2)
        kl_loss = kl_loss / (height * width)
        
        return kl_loss
    
    def _patchembedding_loss(self, student_outputs, teacher_outputs, embedding_weights=[0.1, 0.1, 0.5, 1]):
        # Loss
        emb_loss = 0
        loss_embedding = nn.MSELoss()

        # Patch Embeddings
        student_emb = student_outputs['patch_embeddings']
        teacher_emb = teacher_outputs['patch_embeddings']

        for i in range(len(student_emb)): 
            emb_loss += embedding_weights[i] * loss_embedding(student_emb[i], teacher_emb[i])

        return emb_loss

    def _feature_distillation_loss(self, student_outputs, teacher_outputs, 
                             feature_stage_weights=[1,1,1,1], 
                             lambda_m=1.0, lambda_s_spatial=0.5, lambda_s_channel=0.5):
        """
        Cosine Similarity based Feature Map Distillation Loss

        Args:
            student_outputs
            teacher_outputs
            feature_stage_weights: Weights On Each Stage
            lambda_m: guided MSE loss weight
            lambda_s_spatial: spatial CS loss weight
            lambda_s_channel: channel CS loss weight
        link : https://www.nature.com/articles/s41598-024-69813-6#Equ4 
        """
        # Feature maps
        student_features = student_outputs['feature_map']  # List of 4 feature maps
        teacher_features = teacher_outputs['feature_map']  # List of 4 feature maps
    
        total_featuremap_loss = 0
    
        for i, (student_feat, teacher_feat) in enumerate(zip(student_features, teacher_features)):
            # Feature map channel projection: student -> teacher channel 
            student_feat_proj = self.projection_layers[i](student_feat)  # F^T_ijk
            teacher_feat_detached = teacher_feat.detach()  # F̂^S_ijk
        
            B, C, H, W = student_feat_proj.shape
        
            # === Cosine Similarity ===
        
            # 1. Channel-wise Cosine Similarity (S^C)
            # Feature maps -> [B*H*W, C] reshape
            student_flat_c = student_feat_proj.view(B * H * W, C)  # [BHW, C]
            teacher_flat_c = teacher_feat_detached.view(B * H * W, C)  # [BHW, C]
        
            # Cosine similarity : channel vector similarity
            cos_sim_channel = F.cosine_similarity(student_flat_c, teacher_flat_c, dim=1)  # [BHW]
            cos_sim_channel = cos_sim_channel.view(B, H, W)  # [B, H, W]
        
            # 2. Spatial-wise Cosine Similarity (S^S)
            # Feature maps -> [B*C, H*W] reshape
            student_flat_s = student_feat_proj.view(B * C, H * W)  # [BC, HW]
            teacher_flat_s = teacher_feat_detached.view(B * C, H * W)  # [BC, HW]
        
            # Cosine similarity -> spatial vector similarity
            cos_sim_spatial = F.cosine_similarity(student_flat_s, teacher_flat_s, dim=1)  # [BC]
            cos_sim_spatial = cos_sim_spatial.view(B, C)  # [B, C]
        
            # Guidance Map 생성
            # G = |(1 - S^C(F^T, F̂^S)) · (1 - S^S(F^T, F̂^S))|
            guidance_channel = (1 - cos_sim_channel).abs()  # [B, H, W]
            guidance_spatial = (1 - cos_sim_spatial).abs()   # [B, C]
        
            # Broadcasting
            guidance_spatial_expanded = guidance_spatial.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            guidance_spatial_map = guidance_spatial_expanded.expand(B, C, H, W)  # [B, C, H, W]
        
            # Broadcasting
            guidance_channel_expanded = guidance_channel.unsqueeze(1)  # [B, 1, H, W]
            guidance_channel_map = guidance_channel_expanded.expand(B, C, H, W)  # [B, C, H, W]
        
            # Element-wise guidance map
            guidance_map = guidance_channel_map * guidance_spatial_map  # [B, C, H, W]
        
            # === Feature Distillation Loss ===
        
            # 1. Guided MSE Loss
            mse_diff = (student_feat_proj - teacher_feat_detached) ** 2  # [B, C, H, W]
            guided_mse = (guidance_map * mse_diff).sum() / (C * H * W)
        
            # 2. Spatial CS Loss
            spatial_cs_loss = (1 - cos_sim_spatial).sum() / C  # Average over channels
        
            # 3. Channel CS Loss 
            channel_cs_loss = (1 - cos_sim_channel).sum() / (H * W)  # Average over spatial
        
            # Total loss
            stage_loss = (lambda_m * guided_mse + 
                        lambda_s_spatial * spatial_cs_loss + 
                        lambda_s_channel * channel_cs_loss)
        
            # Weighted Stage loss
            weighted_stage_loss = feature_stage_weights[i] * stage_loss
            total_featuremap_loss += weighted_stage_loss
    
        return total_featuremap_loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = ["patch_embeddings", "feature_map"],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Evaluation step with distillation
        """
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model, inputs, return_outputs=True)
        
        return (eval_loss, pred, label)

class DICELossTrainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        weight = torch.ones(20)
        weight[0] = 2.5959737
        weight[1] = 6.741505
        weight[2] = 3.5353868
        weight[3] = 9.866315
        weight[4] = 9.690922
        weight[5] = 9.369371
        weight[6] = 10.289124 
        weight[7] = 9.953209
        weight[8] = 4.3098087
        weight[9] = 9.490392
        weight[10] = 7.674411
        weight[11] = 9.396925	
        weight[12] = 10.347794 	
        weight[13] = 6.3928986
        weight[14] = 10.226673 	
        weight[15] = 10.241072	
        weight[16] = 10.28059
        weight[17] = 10.396977
        weight[18] = 10.05567	
        weight[19] = 0
        weight = weight.to(self.device)     

        self.criterion = DiceFocalLoss(num_classes=20, weight=weight, ignore_index=19)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        image = inputs['pixel_values'].to(self.device)
        label = inputs['labels'].to(self.device)

        output = model(image)

        loss = self.criterion(output['logits'], label)

        if return_outputs:
            return loss, output, label
        
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict,
            prediction_loss_only: bool,
            ignore_keys=None,
    ):
        model.eval()
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model, inputs, return_outputs=True)
        return (eval_loss, pred, label)
    
class DiceFocalLossTrainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        weight = torch.ones(20)
        weight[0] = 2.5959737
        weight[1] = 6.741505
        weight[2] = 3.5353868
        weight[3] = 9.866315
        weight[4] = 9.690922
        weight[5] = 9.369371
        weight[6] = 10.289124 
        weight[7] = 9.953209
        weight[8] = 4.3098087
        weight[9] = 9.490392
        weight[10] = 7.674411
        weight[11] = 9.396925	
        weight[12] = 10.347794 	
        weight[13] = 6.3928986
        weight[14] = 10.226673 	
        weight[15] = 10.241072	
        weight[16] = 10.28059
        weight[17] = 10.396977
        weight[18] = 10.05567	
        weight[19] = 0
        weight = weight.to(self.device)     

        self.criterion = DiceFocalLoss_updated(num_classes=20, weight=weight, ignore_index=19)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        image = inputs['pixel_values'].to(self.device)
        label = inputs['labels'].to(self.device)

        output = model(image)

        loss = self.criterion(output['logits'], label)

        if return_outputs:
            return loss, output, label
        
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict,
            prediction_loss_only: bool,
            ignore_keys=None,
    ):
        model.eval()
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model, inputs, return_outputs=True)
        return (eval_loss, pred, label)
    
class TanhTrainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.loss_fn = CrossEntropyLoss2d()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    def compute_loss(self, model, inputs, num_items_in_batch = None, return_outputs=False):
        image = inputs['pixel_values'].to(self.device)
        label = inputs['labels'].to(self.device)
        # Forward pass
        output = model(image) # output is a dictionary (feature map :  ,logits:torch.Size([2, 20, 128, 256]), patch embeddings: torch.Size([2, 32768, 32], torch.Size([2, 8192, 64])...))
        # Compute loss
        loss = self.loss_fn(output, label)

        if return_outputs:
            return loss, output, label
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = ["patch_embeddings", "feature_map"],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: # Optional은 Tensor 또는 None이 올 수 있다는 뜻 
        
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model,inputs,return_outputs = True)
        
        return (eval_loss,pred,label)

# Final Trainer
# Hard loss + Patch Embedding loss (1.0) + Feature Map Distillation loss(Cosine Similarity loss) (1.0)
class KDTrainer(Trainer):
    def __init__(self, teacher_model, temperature=1.0, alpha=0.0, beta=1.0, gamma=1.0, 
                 teacher_channels=[64, 128, 320, 512], student_channels=[32, 64, 160, 256], **kwds): # -> default channel set to b2, b0 channels
        super().__init__(**kwds)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Teacher model load for distillation
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval() 
        
        # Distillation parameters
        self.temperature = temperature
        self.alpha = alpha  # distillation loss weight (KL)
        self.beta = beta    # PE Loss weight
        self.gamma = gamma  # Feature map distillation weight
        
        # Loss function -> Target Loss
        self.loss_fn = CrossEntropyLoss2d()
        
        # Channel Informations of Student and Teacher
        self.teacher_channels = teacher_channels
        self.student_channels = student_channels
        
        # Channel projection layers -> 1*1 Convolution
        self.projection_layers = nn.ModuleList()
        for t_ch, s_ch in zip(teacher_channels, student_channels):
            if t_ch != s_ch:
                proj = nn.Sequential(
                    nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(t_ch),
                    nn.ReLU()
                )
            else:
                proj = nn.Identity()
            self.projection_layers.append(proj)
        
        # Projection layers for each stage to device
        self.projection_layers.to(self.device)
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        compute loss including label loss + distillation losses
        """
        image = inputs['pixel_values'].to(self.device)
        label = inputs['labels'].to(self.device)
        
        # student forward pass
        student_output = model(image)
        
        # teacher forward pass => eval mode
        with torch.no_grad():
            teacher_output = self.teacher_model(image)
        
        # Target loss 
        hard_loss = self.loss_fn(student_output, label)
        
        # Pixelwise Distillation Loss
        soft_loss = self._distillation_loss(student_output, teacher_output)

        # PE Loss
        pe_loss = self._patchembedding_loss(student_output, teacher_output)
        
        # Feature Map Distillation Loss
        feature_loss = self._feature_distillation_loss(student_output, teacher_output)
        
        # Combined loss
        total_loss = hard_loss + self.alpha * soft_loss + self.beta * pe_loss + self.gamma * feature_loss
        
        if return_outputs:
            return total_loss, student_output, label
        return total_loss
    
    def _distillation_loss(self, student_logits, teacher_logits):
        """
        Segmentation Knowledge Distillation Loss
        Ver1. total loss = label loss + pixel level KL divergence distillation loss
        """
        # Logits
        student_logits = student_logits['logits']
        teacher_logits = teacher_logits['logits']

        # Logits shape: [batch_size, num_classes, height, width]
        batch_size, num_classes, height, width = student_logits.shape
        
        # Flatten spatial dimensions for easier computation
        student_flat = student_logits.view(batch_size, num_classes, -1)  # [B, C, H*W]
        teacher_flat = teacher_logits.view(batch_size, num_classes, -1)  # [B, C, H*W]
        
        # Temperature scaling and softmax
        student_soft = F.softmax(student_flat / self.temperature, dim=1)  # [B, C, H*W]
        teacher_soft = F.softmax(teacher_flat / self.temperature, dim=1)  # [B, C, H*W]
        
        # KL Divergence Loss
        kl_loss = F.kl_div(F.log_softmax(student_flat / self.temperature, dim=1), teacher_soft, reduction='batchmean')
        kl_loss = kl_loss * (self.temperature ** 2)
        kl_loss = kl_loss / (height * width)
        
        return kl_loss
    
    def _patchembedding_loss(self, student_outputs, teacher_outputs, embedding_weights=[0.1, 0.1, 0.5, 1]):
        # Loss
        emb_loss = 0
        loss_embedding = nn.MSELoss()

        # Patch Embeddings
        student_emb = student_outputs['patch_embeddings']
        teacher_emb = teacher_outputs['patch_embeddings']

        for i in range(len(student_emb)): 
            emb_loss += embedding_weights[i] * loss_embedding(student_emb[i], teacher_emb[i])

        return emb_loss

    def _feature_distillation_loss(self, student_outputs, teacher_outputs, 
                             feature_stage_weights=[1,1,1,1], 
                             lambda_m=1.0, lambda_s_spatial=0.5, lambda_s_channel=0.5):
        """
        Cosine Similarity based Feature Map Distillation Loss

        Args:
            student_outputs
            teacher_outputs
            feature_stage_weights: Weights On Each Stage
            lambda_m: guided MSE loss weight
            lambda_s_spatial: spatial CS loss weight
            lambda_s_channel: channel CS loss weight
        link : https://www.nature.com/articles/s41598-024-69813-6#Equ4 
        """
        # Feature maps
        student_features = student_outputs['feature_map']  # List of 4 feature maps
        teacher_features = teacher_outputs['feature_map']  # List of 4 feature maps
    
        total_featuremap_loss = 0
    
        for i, (student_feat, teacher_feat) in enumerate(zip(student_features, teacher_features)):
            # Feature map channel projection: student -> teacher channel 
            student_feat_proj = self.projection_layers[i](student_feat)  # F^T_ijk
            teacher_feat_detached = teacher_feat.detach()  # F̂^S_ijk
        
            B, C, H, W = student_feat_proj.shape
        
            # === Cosine Similarity ===
        
            # 1. Channel-wise Cosine Similarity (S^C)
            # Feature maps -> [B*H*W, C] reshape
            student_flat_c = student_feat_proj.view(B * H * W, C)  # [BHW, C]
            teacher_flat_c = teacher_feat_detached.view(B * H * W, C)  # [BHW, C]
        
            # Cosine similarity : channel vector similarity
            cos_sim_channel = F.cosine_similarity(student_flat_c, teacher_flat_c, dim=1)  # [BHW]
            cos_sim_channel = cos_sim_channel.view(B, H, W)  # [B, H, W]
        
            # 2. Spatial-wise Cosine Similarity (S^S)
            # Feature maps -> [B*C, H*W] reshape
            student_flat_s = student_feat_proj.view(B * C, H * W)  # [BC, HW]
            teacher_flat_s = teacher_feat_detached.view(B * C, H * W)  # [BC, HW]
        
            # Cosine similarity -> spatial vector similarity
            cos_sim_spatial = F.cosine_similarity(student_flat_s, teacher_flat_s, dim=1)  # [BC]
            cos_sim_spatial = cos_sim_spatial.view(B, C)  # [B, C]
        
            # Guidance Map 생성
            # G = |(1 - S^C(F^T, F̂^S)) · (1 - S^S(F^T, F̂^S))|
            guidance_channel = (1 - cos_sim_channel).abs()  # [B, H, W]
            guidance_spatial = (1 - cos_sim_spatial).abs()   # [B, C]
        
            # Broadcasting
            guidance_spatial_expanded = guidance_spatial.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            guidance_spatial_map = guidance_spatial_expanded.expand(B, C, H, W)  # [B, C, H, W]
        
            # Broadcasting
            guidance_channel_expanded = guidance_channel.unsqueeze(1)  # [B, 1, H, W]
            guidance_channel_map = guidance_channel_expanded.expand(B, C, H, W)  # [B, C, H, W]
        
            # Element-wise guidance map
            guidance_map = guidance_channel_map * guidance_spatial_map  # [B, C, H, W]
        
            # === Feature Distillation Loss ===
        
            # 1. Guided MSE Loss
            mse_diff = (student_feat_proj - teacher_feat_detached) ** 2  # [B, C, H, W]
            guided_mse = (guidance_map * mse_diff).sum() / (C * H * W)
        
            # 2. Spatial CS Loss
            spatial_cs_loss = (1 - cos_sim_spatial).sum() / C  # Average over channels
        
            # 3. Channel CS Loss 
            channel_cs_loss = (1 - cos_sim_channel).sum() / (H * W)  # Average over spatial
        
            # Total loss
            stage_loss = (lambda_m * guided_mse + 
                        lambda_s_spatial * spatial_cs_loss + 
                        lambda_s_channel * channel_cs_loss)
        
            # Weighted Stage loss
            weighted_stage_loss = feature_stage_weights[i] * stage_loss
            total_featuremap_loss += weighted_stage_loss
    
        return total_featuremap_loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = ["patch_embeddings", "feature_map"],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Evaluation step with distillation
        """
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model, inputs, return_outputs=True)
        
        return (eval_loss, pred, label)

# Hard loss + Patch Embedding loss (1.0) + Feature Map Distillation loss(Cosine Similarity loss) (1.0) + KL(1.0) -> temperature set to 2.0
# Take balance between KL and PE, maintain the ratio of Feature map cosine similarity loss
class EffisegBalancedTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=1.0, beta=1.0, gamma=1.0, 
                 teacher_channels=[64, 128, 320, 512], student_channels=[32, 64, 160, 256], **kwds): # -> default channel set to b2, b0 channels
        super().__init__(**kwds)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Teacher model load for distillation
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval() 
        
        # Distillation parameters
        self.temperature = temperature
        self.alpha = alpha  # distillation loss weight (KL)
        self.beta = beta    # PE Loss weight
        self.gamma = gamma  # Feature map distillation weight
        
        # Loss function -> Target Loss
        self.loss_fn = CrossEntropyLoss2d()
        
        # Channel Informations of Student and Teacher
        self.teacher_channels = teacher_channels
        self.student_channels = student_channels
        
        # Channel projection layers -> 1*1 Convolution
        self.projection_layers = nn.ModuleList()
        for t_ch, s_ch in zip(teacher_channels, student_channels):
            if t_ch != s_ch:
                proj = nn.Sequential(
                    nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(t_ch),
                    nn.ReLU()
                )
            else:
                proj = nn.Identity()
            self.projection_layers.append(proj)
        
        # Projection layers for each stage to device
        self.projection_layers.to(self.device)
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        compute loss including label loss + distillation losses
        """
        image = inputs['pixel_values'].to(self.device)
        label = inputs['labels'].to(self.device)
        
        # student forward pass
        student_output = model(image)
        
        # teacher forward pass => eval mode
        with torch.no_grad():
            teacher_output = self.teacher_model(image)
        
        # Target loss 
        hard_loss = self.loss_fn(student_output, label)
        
        # Pixelwise Distillation Loss
        soft_loss = self._distillation_loss(student_output, teacher_output)

        # PE Loss
        pe_loss = self._patchembedding_loss(student_output, teacher_output)
        
        # Feature Map Distillation Loss
        feature_loss = self._feature_distillation_loss(student_output, teacher_output)
        
        # Combined loss
        total_loss = hard_loss + self.alpha * soft_loss + self.beta * pe_loss + self.gamma * feature_loss
        
        if return_outputs:
            return total_loss, student_output, label
        return total_loss
    
    def _distillation_loss(self, student_logits, teacher_logits):
        """
        Segmentation Knowledge Distillation Loss
        Ver1. total loss = label loss + pixel level KL divergence distillation loss
        """
        # Logits
        student_logits = student_logits['logits']
        teacher_logits = teacher_logits['logits']

        # Logits shape: [batch_size, num_classes, height, width]
        batch_size, num_classes, height, width = student_logits.shape
        
        # Flatten spatial dimensions for easier computation
        student_flat = student_logits.view(batch_size, num_classes, -1)  # [B, C, H*W]
        teacher_flat = teacher_logits.view(batch_size, num_classes, -1)  # [B, C, H*W]
        
        # Temperature scaling and softmax
        student_soft = F.softmax(student_flat / self.temperature, dim=1)  # [B, C, H*W]
        teacher_soft = F.softmax(teacher_flat / self.temperature, dim=1)  # [B, C, H*W]
        
        # KL Divergence Loss
        kl_loss = F.kl_div(F.log_softmax(student_flat / self.temperature, dim=1), teacher_soft, reduction='batchmean')
        kl_loss = kl_loss * (self.temperature ** 2)
        kl_loss = kl_loss / (height * width)
        
        return kl_loss
    
    def _patchembedding_loss(self, student_outputs, teacher_outputs, embedding_weights=[0.1, 0.1, 0.5, 1]):
        # Loss
        emb_loss = 0
        loss_embedding = nn.MSELoss()

        # Patch Embeddings
        student_emb = student_outputs['patch_embeddings']
        teacher_emb = teacher_outputs['patch_embeddings']

        for i in range(len(student_emb)): 
            emb_loss += embedding_weights[i] * loss_embedding(student_emb[i], teacher_emb[i])

        return emb_loss

    def _feature_distillation_loss(self, student_outputs, teacher_outputs, 
                             feature_stage_weights=[1,1,1,1], 
                             lambda_m=1.0, lambda_s_spatial=0.5, lambda_s_channel=0.5):
        """
        Cosine Similarity based Feature Map Distillation Loss

        Args:
            student_outputs
            teacher_outputs
            feature_stage_weights: Weights On Each Stage
            lambda_m: guided MSE loss weight
            lambda_s_spatial: spatial CS loss weight
            lambda_s_channel: channel CS loss weight
        link : https://www.nature.com/articles/s41598-024-69813-6#Equ4 
        """
        # Feature maps
        student_features = student_outputs['feature_map']  # List of 4 feature maps
        teacher_features = teacher_outputs['feature_map']  # List of 4 feature maps
    
        total_featuremap_loss = 0
    
        for i, (student_feat, teacher_feat) in enumerate(zip(student_features, teacher_features)):
            # Feature map channel projection: student -> teacher channel 
            student_feat_proj = self.projection_layers[i](student_feat)  # F^T_ijk
            teacher_feat_detached = teacher_feat.detach()  # F̂^S_ijk
        
            B, C, H, W = student_feat_proj.shape
        
            # === Cosine Similarity ===
        
            # 1. Channel-wise Cosine Similarity (S^C)
            # Feature maps -> [B*H*W, C] reshape
            student_flat_c = student_feat_proj.view(B * H * W, C)  # [BHW, C]
            teacher_flat_c = teacher_feat_detached.view(B * H * W, C)  # [BHW, C]
        
            # Cosine similarity : channel vector similarity
            cos_sim_channel = F.cosine_similarity(student_flat_c, teacher_flat_c, dim=1)  # [BHW]
            cos_sim_channel = cos_sim_channel.view(B, H, W)  # [B, H, W]
        
            # 2. Spatial-wise Cosine Similarity (S^S)
            # Feature maps -> [B*C, H*W] reshape
            student_flat_s = student_feat_proj.view(B * C, H * W)  # [BC, HW]
            teacher_flat_s = teacher_feat_detached.view(B * C, H * W)  # [BC, HW]
        
            # Cosine similarity -> spatial vector similarity
            cos_sim_spatial = F.cosine_similarity(student_flat_s, teacher_flat_s, dim=1)  # [BC]
            cos_sim_spatial = cos_sim_spatial.view(B, C)  # [B, C]
        
            # Guidance Map 생성
            # G = |(1 - S^C(F^T, F̂^S)) · (1 - S^S(F^T, F̂^S))|
            guidance_channel = (1 - cos_sim_channel).abs()  # [B, H, W]
            guidance_spatial = (1 - cos_sim_spatial).abs()   # [B, C]
        
            # Broadcasting
            guidance_spatial_expanded = guidance_spatial.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            guidance_spatial_map = guidance_spatial_expanded.expand(B, C, H, W)  # [B, C, H, W]
        
            # Broadcasting
            guidance_channel_expanded = guidance_channel.unsqueeze(1)  # [B, 1, H, W]
            guidance_channel_map = guidance_channel_expanded.expand(B, C, H, W)  # [B, C, H, W]
        
            # Element-wise guidance map
            guidance_map = guidance_channel_map * guidance_spatial_map  # [B, C, H, W]
        
            # === Feature Distillation Loss ===
        
            # 1. Guided MSE Loss
            mse_diff = (student_feat_proj - teacher_feat_detached) ** 2  # [B, C, H, W]
            guided_mse = (guidance_map * mse_diff).sum() / (C * H * W)
        
            # 2. Spatial CS Loss
            spatial_cs_loss = (1 - cos_sim_spatial).sum() / C  # Average over channels
        
            # 3. Channel CS Loss 
            channel_cs_loss = (1 - cos_sim_channel).sum() / (H * W)  # Average over spatial
        
            # Total loss
            stage_loss = (lambda_m * guided_mse + 
                        lambda_s_spatial * spatial_cs_loss + 
                        lambda_s_channel * channel_cs_loss)
        
            # Weighted Stage loss
            weighted_stage_loss = feature_stage_weights[i] * stage_loss
            total_featuremap_loss += weighted_stage_loss
    
        return total_featuremap_loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = ["patch_embeddings", "feature_map"],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Evaluation step with distillation
        """
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model, inputs, return_outputs=True)
        
        return (eval_loss, pred, label)