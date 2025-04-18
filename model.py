import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SparseAttentionBlock(nn.Module):
    """
    Sparse Attention Block for the Global Feature Element-based Attention Network (GFE-AN)
    This module selectively emphasizes feature elements meaningful for facial expression
    and suppresses those unrelated to facial expression.
    """
    def __init__(self, in_features, reduction_ratio=16):
        super(SparseAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction_ratio, in_features, bias=False),
            nn.Sigmoid()
        )
        # L1 regularization to enforce sparsity
        self.l1_strength = 0.01

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        # Store attention weights for visualization and regularization
        self.attention_weights = y.squeeze()
        
        return x * y.expand_as(x), self.attention_weights

class LocalFeatureAttentionBlock(nn.Module):
    """
    Local Feature Attention Block for the Multi-Feature Fusion-based Attention Network (MFF-AN)
    This module extracts rich semantic information from different representation sub-spaces.
    """
    def __init__(self, in_channels, out_channels, reduction_ratio=8):
        super(LocalFeatureAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction_ratio, out_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Channel attention
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_out = self.sigmoid(avg_out + max_out)
        x = x * channel_out
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.spatial_conv(spatial_out)
        spatial_out = self.sigmoid(spatial_out)
        
        x = x * spatial_out
        
        return x

class HighLevelFeatureExtractor(nn.Module):
    """
    High-level feature extractor using a pre-trained CNN backbone
    """
    def __init__(self, backbone='resnet18', pretrained=True):
        super(HighLevelFeatureExtractor, self).__init__()
        
        # Load pre-trained backbone
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the classification layer
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        
    def forward(self, x):
        return self.features(x)

class GFE_AN(nn.Module):
    """
    Global Feature Element-based Attention Network (GFE-AN)
    This network applies attention to the final feature vector to selectively
    emphasize informative feature elements and suppress those irrelevant to facial expression.
    """
    def __init__(self, feature_dim, num_classes, dropout_rate=0.5):
        super(GFE_AN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sparse_attention = SparseAttentionBlock(feature_dim)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
    def forward(self, x):
        x, attention_weights = self.sparse_attention(x)
        x = self.avg_pool(x)
        features = x.view(x.size(0), -1)
        out = self.fc(features)
        return out, features, attention_weights

class MFF_AN(nn.Module):
    """
    Multi-Feature Fusion-based Attention Network (MFF-AN)
    This network extracts various facial features from multiple sub-networks
    to make the model insensitive to occlusion and pose variation.
    """
    def __init__(self, feature_dim, num_classes, num_subnetworks=3, dropout_rate=0.5):
        super(MFF_AN, self).__init__()
        self.num_subnetworks = num_subnetworks
        
        # Create multiple sub-networks with local feature attention
        self.sub_networks = nn.ModuleList([
            LocalFeatureAttentionBlock(feature_dim, feature_dim // 2)
            for _ in range(num_subnetworks)
        ])
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fusion mechanism
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim // 2 * num_subnetworks, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
    def forward(self, x):
        # Process through each sub-network
        sub_features = []
        for i in range(self.num_subnetworks):
            sub_feat = self.sub_networks[i](x)
            sub_feat = self.avg_pool(sub_feat)
            sub_feat = sub_feat.view(sub_feat.size(0), -1)
            sub_features.append(sub_feat)
        
        # Concatenate all sub-network features
        concat_features = torch.cat(sub_features, dim=1)
        
        # Fusion
        out = self.fusion(concat_features)
        
        return out, concat_features

class DSAN(nn.Module):
    """
    Dual Stream Attention Network (DSAN) for facial expression recognition
    Consists of GFE-AN and MFF-AN to handle occlusion and head pose variation.
    """
    def __init__(self, num_classes=7, backbone='resnet18', pretrained=True):
        super(DSAN, self).__init__()
        
        # High-level feature extractor
        self.feature_extractor = HighLevelFeatureExtractor(backbone, pretrained)
        feature_dim = self.feature_extractor.feature_dim
        
        # GFE-AN branch
        self.gfe_an = GFE_AN(feature_dim, num_classes)
        
        # MFF-AN branch
        self.mff_an = MFF_AN(feature_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Count parameters
        self.count_parameters()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract high-level features
        features = self.feature_extractor(x)
        
        # Global Feature Element-based Attention Network
        gfe_out, gfe_features, attention_weights = self.gfe_an(features)
        
        # Multi-Feature Fusion-based Attention Network
        mff_out, mff_features = self.mff_an(features)
        
        # Fusion of both outputs (weighted sum)
        if self.training:
            return gfe_out, mff_out, attention_weights
        else:
            # During inference, combine the predictions
            combined_out = (gfe_out + mff_out) / 2
            return combined_out, attention_weights
            
    def count_parameters(self):
        """Count and print the number of parameters in the model"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
        return total_params

class FeatureRecalibrationLoss(nn.Module):
    """
    Feature Recalibration Loss designed to increase inter-class distance
    and decrease intra-class distance for facial expression features
    """
    def __init__(self, lambda_val=0.1):
        super(FeatureRecalibrationLoss, self).__init__()
        self.lambda_val = lambda_val
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, features, targets, attention_weights=None):
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(outputs, targets)
        
        # Center loss for feature clustering
        batch_size = features.size(0)
        classes = torch.unique(targets)
        center_loss = 0.0
        
        # Calculate centers for each class
        centers = {}
        for c in classes:
            class_features = features[targets == c]
            if len(class_features) > 0:
                centers[c.item()] = torch.mean(class_features, dim=0)
        
        # Calculate intra-class distance (to own center)
        for i in range(batch_size):
            if targets[i].item() in centers:
                center = centers[targets[i].item()]
                center_loss += torch.sum((features[i] - center) ** 2)
        
        if batch_size > 0:
            center_loss /= batch_size
        
        # L1 regularization on attention weights (sparsity)
        l1_loss = 0.0
        if attention_weights is not None:
            l1_loss = torch.mean(torch.abs(attention_weights))
        
        # Combined loss
        total_loss = ce_loss + self.lambda_val * center_loss + 0.01 * l1_loss
        
        return total_loss, ce_loss, center_loss, l1_loss
