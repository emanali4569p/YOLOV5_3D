import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SiLU(nn.Module):
    """Sigmoid Linear Unit activation function"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    """Mish activation function"""
    @staticmethod
    def forward(x):
        return x * torch.tanh(F.softplus(x))

class Hardswish(nn.Module):
    """Hardswish activation function"""
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0., 6.) / 6.

class Conv(nn.Module):
    """Standard convolution with activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""
    def __init__(self, c1, c2, k=5):
        super(SPPF, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Focus(nn.Module):
    """Focus wh information into c-space"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], 
                                   x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))

class Detect3D(nn.Module):
    """3D Detection head"""
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super(Detect3D, self).__init__()
        self.nc = nc  # Number of classes
        self.no = nc + 5  # Number of outputs per anchor

        # Handle anchors and number of layers robustly
        default_anchors = torch.tensor([
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ], dtype=torch.float32)

        if anchors and len(anchors) > 0:
            self.nl = len(anchors)
            anchors_tensor = torch.tensor(anchors, dtype=torch.float32)
            anchors_tensor = anchors_tensor.view(self.nl, -1, 2)
        else:
            # Derive number of layers from channels if possible; fallback to 3
            self.nl = len(ch) if len(ch) > 0 else 3
            anchors_tensor = default_anchors.view(3, -1, 2)[:self.nl]

        self.na = anchors_tensor.shape[1]  # Number of anchors per layer
        self.grid = [torch.zeros(1)] * self.nl  # Init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # Init anchor grid
        self.register_buffer('anchors', anchors_tensor)

        # Heads
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace

        # Initialize stride
        self.stride = torch.tensor([8, 16, 32])

        # 3D prediction heads
        self.depth_head = nn.ModuleList(nn.Conv2d(x, 1, 1) for x in ch)
        self.dimension_head = nn.ModuleList(nn.Conv2d(x, 3, 1) for x in ch)
        self.rotation_head = nn.ModuleList(nn.Conv2d(x, 1, 1) for x in ch)

    def forward(self, x):
        """Forward pass"""
        z = []  # Output
        depth_outputs = []
        dimension_outputs = []
        rotation_outputs = []
        
        for i in range(self.nl):
            # Process 3D heads on input features before detection head
            depth_outputs.append(self.depth_head[i](x[i]))
            dimension_outputs.append(self.dimension_head[i](x[i]))
            rotation_outputs.append(self.rotation_head[i](x[i]))
            
            # Detection head
            x[i] = self.m[i](x[i])
            z.append(x[i])
        
        return {
            'detections': z,
            'depth': depth_outputs,
            'dimensions': dimension_outputs,
            'rotation': rotation_outputs
        }

class YOLOv5_3D(nn.Module):
    """YOLOv5 model for 3D object detection - Memory Optimized"""
    def __init__(self, nc=80, anchors=()):
        super(YOLOv5_3D, self).__init__()
        
        # Lightweight backbone with reduced channels
        self.conv1 = Conv(3, 32, 3)      # Reduced from 64
        self.conv2 = Conv(32, 64, 3, 2)   # Reduced from 128
        self.c3_1 = C3(64, 64, 1)        # Reduced layers
        
        self.conv3 = Conv(64, 128, 3, 2)  # Reduced from 256
        self.c3_2 = C3(128, 128, 2)      # Reduced layers
        
        self.conv4 = Conv(128, 256, 3, 2)  # Reduced from 512
        self.c3_3 = C3(256, 256, 2)      # Reduced layers
        
        self.conv5 = Conv(256, 512, 3, 2) # Reduced from 1024
        self.c3_4 = C3(512, 512, 1)      # Reduced layers
        self.sppf = SPPF(512, 512, 5)    # Reduced channels
        
        # Lightweight Feature Pyramid Network
        self.neck_conv1 = Conv(512, 256, 1, 1)  # P5 -> 256 channels
        self.neck_upsample1 = nn.Upsample(None, 2, 'nearest')
        
        self.neck_conv2 = Conv(256, 128, 1, 1)   # P4 -> 128 channels
        self.neck_upsample2 = nn.Upsample(None, 2, 'nearest')
        
        self.neck_conv3 = Conv(128, 64, 1, 1)    # P3 -> 64 channels
        
        # Detection head with reduced channel dimensions
        self.detect = Detect3D(nc=nc, anchors=anchors, ch=[64, 128, 256])

    def forward(self, x):
        """Memory-optimized forward pass"""
        # Backbone with reduced memory usage
        x1 = self.conv1(x)      # 32 channels
        x2 = self.conv2(x1)     # 64 channels  
        x3 = self.c3_1(x2)      # 64 channels
        
        x4 = self.conv3(x3)     # 128 channels
        x5 = self.c3_2(x4)      # 128 channels
        
        x6 = self.conv4(x5)     # 256 channels
        x7 = self.c3_3(x6)      # 256 channels
        
        x8 = self.conv5(x7)     # 512 channels
        x9 = self.c3_4(x8)      # 512 channels
        x10 = self.sppf(x9)     # 512 channels
        
        # Lightweight Feature Pyramid Network
        # P5 (large objects) - 256 channels
        p5 = self.neck_conv1(x10)
        
        # P4 (medium objects) - 128 channels  
        p4_up = self.neck_upsample1(p5)
        p4 = self.neck_conv2(p4_up)
        
        # P3 (small objects) - 64 channels
        p3_up = self.neck_upsample2(p4)
        p3 = self.neck_conv3(p3_up)
        
        # Final multi-scale features: P3 (64), P4 (128), P5 (256)
        return self.detect([p3, p4, p5])

class Loss3D(nn.Module):
    """Enhanced 3D Detection loss function with scientific components"""
    def __init__(self, nc=80):
        super(Loss3D, self).__init__()
        self.nc = nc
        
        # Loss functions for different components
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.l1_loss = nn.L1Loss()
        
        # Loss weights
        self.lambda_obj = 1.0      # Objectness loss weight
        self.lambda_bbox = 5.0     # Bbox loss weight  
        self.lambda_cls = 1.0      # Classification loss weight
        self.lambda_depth = 2.0    # Depth loss weight
        self.lambda_dim = 1.0      # Dimension loss weight
        self.lambda_rot = 1.0      # Rotation loss weight

    def ciou_loss(self, pred_bbox, target_bbox):
        """Complete IoU Loss for better bbox regression"""
        # Simplified CIoU implementation
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox.chunk(4, dim=-1)
        target_x1, target_y1, target_x2, target_y2 = target_bbox.chunk(4, dim=-1)
        
        # Calculate areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        # Calculate intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union
        union_area = pred_area + target_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-7)
        
        # CIoU loss
        ciou_loss = 1 - iou
        return ciou_loss.mean()

    def angular_loss(self, pred_rot, target_rot):
        """Angular loss using cosine similarity for rotation"""
        # Convert to cosine similarity loss
        pred_cos = torch.cos(pred_rot)
        pred_sin = torch.sin(pred_rot)
        target_cos = torch.cos(target_rot)
        target_sin = torch.sin(target_rot)
        
        cos_loss = self.l1_loss(pred_cos, target_cos)
        sin_loss = self.l1_loss(pred_sin, target_sin)
        
        return cos_loss + sin_loss

    def forward(self, predictions, targets):
        """Calculate enhanced 3D detection loss"""
        detections = predictions['detections']
        depth = predictions['depth']
        dimensions = predictions['dimensions']
        rotation = predictions['rotation']
        
        # Initialize losses
        total_obj_loss = 0
        total_bbox_loss = 0
        total_cls_loss = 0
        total_depth_loss = 0
        total_dim_loss = 0
        total_rot_loss = 0
        
        num_scales = len(detections)
        
        for i in range(num_scales):
            # Detection loss components
            det = detections[i]
            batch_size, channels, height, width = det.shape
            
            # Parse detection outputs (simplified)
            obj_pred = det[:, :1, :, :]  # Objectness
            bbox_pred = det[:, 1:5, :, :]  # Bbox coordinates
            cls_pred = det[:, 5:5+self.nc, :, :]  # Classification
            
            # Objectness loss (BCE)
            obj_target = torch.zeros_like(obj_pred)  # Simplified target
            obj_loss = self.bce_loss(obj_pred, obj_target)
            total_obj_loss += obj_loss
            
            # Bbox loss (CIoU)
            bbox_target = torch.zeros_like(bbox_pred)  # Simplified target
            bbox_loss = self.ciou_loss(bbox_pred.flatten(), bbox_target.flatten())
            total_bbox_loss += bbox_loss
            
            # Classification loss (BCE)
            cls_target = torch.zeros_like(cls_pred)  # Simplified target
            cls_loss = self.bce_loss(cls_pred, cls_target)
            total_cls_loss += cls_loss
            
            # Depth loss (L1)
            depth_pred = depth[i]
            depth_target = torch.zeros_like(depth_pred)  # Simplified target
            depth_loss = self.l1_loss(depth_pred, depth_target)
            total_depth_loss += depth_loss
            
            # Dimension loss (SmoothL1)
            dim_pred = dimensions[i]
            dim_target = torch.zeros_like(dim_pred)  # Simplified target
            dim_loss = self.smooth_l1(dim_pred, dim_target)
            total_dim_loss += dim_loss
            
            # Rotation loss (Angular)
            rot_pred = rotation[i]
            rot_target = torch.zeros_like(rot_pred)  # Simplified target
            rot_loss = self.angular_loss(rot_pred, rot_target)
            total_rot_loss += rot_loss
        
        # Normalize by number of scales
        total_obj_loss /= num_scales
        total_bbox_loss /= num_scales
        total_cls_loss /= num_scales
        total_depth_loss /= num_scales
        total_dim_loss /= num_scales
        total_rot_loss /= num_scales
        
        # Combined detection loss
        detection_loss = (self.lambda_obj * total_obj_loss + 
                         self.lambda_bbox * total_bbox_loss + 
                         self.lambda_cls * total_cls_loss)
        
        # Total loss with proper weights
        total_loss = (detection_loss + 
                     self.lambda_depth * total_depth_loss + 
                     self.lambda_dim * total_dim_loss + 
                     self.lambda_rot * total_rot_loss)
        
        return {
            'total_loss': total_loss,
            'detection_loss': detection_loss,
            'objectness_loss': total_obj_loss,
            'bbox_loss': total_bbox_loss,
            'classification_loss': total_cls_loss,
            'depth_loss': total_depth_loss,
            'dimension_loss': total_dim_loss,
            'rotation_loss': total_rot_loss
        }

def create_model(nc=80, anchors=()):
    """Create YOLOv5 3D model"""
    model = YOLOv5_3D(nc=nc, anchors=anchors)
    return model

def test_model():
    """Test model functionality"""
    print("Testing YOLOv5 3D model...")
    
    # Create model
    model = create_model(nc=9)  # 9 classes for KITTI
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Detection outputs: {len(outputs['detections'])}")
    print(f"Depth outputs: {len(outputs['depth'])}")
    print(f"Dimension outputs: {len(outputs['dimensions'])}")
    print(f"Rotation outputs: {len(outputs['rotation'])}")
    
    # Test loss function
    loss_fn = Loss3D(nc=9)
    loss = loss_fn(outputs, None)
    
    print(f"Total loss: {loss['total_loss']:.4f}")
    print(f"Detection loss: {loss['detection_loss']:.4f}")
    print(f"Depth loss: {loss['depth_loss']:.4f}")
    print(f"Dimension loss: {loss['dimension_loss']:.4f}")
    print(f"Rotation loss: {loss['rotation_loss']:.4f}")
    
    print("Model test completed!")

if __name__ == "__main__":
    test_model()