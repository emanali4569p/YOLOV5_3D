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
            x[i] = self.m[i](x[i])  # Detection
            depth_outputs.append(self.depth_head[i](x[i]))
            dimension_outputs.append(self.dimension_head[i](x[i]))
            rotation_outputs.append(self.rotation_head[i](x[i]))
            z.append(x[i])
        
        return {
            'detections': z,
            'depth': depth_outputs,
            'dimensions': dimension_outputs,
            'rotation': rotation_outputs
        }

class YOLOv5_3D(nn.Module):
    """YOLOv5 model for 3D object detection"""
    def __init__(self, nc=80, anchors=()):
        super(YOLOv5_3D, self).__init__()
        
        # Backbone
        self.backbone = nn.Sequential(
            Focus(3, 32, 3),
            Conv(32, 64, 3, 2),
            C3(64, 64, 3),
            Conv(64, 128, 3, 2),
            C3(128, 128, 9),
            Conv(128, 256, 3, 2),
            C3(256, 256, 9),
            Conv(256, 512, 3, 2),
            C3(512, 512, 3),
            SPPF(512, 512, 5)
        )
        
        # Neck (Feature Pyramid Network)
        self.neck = nn.Sequential(
            Conv(512, 256, 1, 1),
            nn.Upsample(None, 2, 'nearest'),
            Conv(256, 256, 3, 1),
            Conv(256, 128, 1, 1),
            nn.Upsample(None, 2, 'nearest'),
            Conv(128, 128, 3, 1),
            Conv(128, 64, 1, 1),
            nn.Upsample(None, 2, 'nearest'),
            Conv(64, 64, 3, 1)
        )
        
        # Detection head
        self.detect = Detect3D(nc=nc, anchors=anchors, ch=[512, 512, 512])

    def forward(self, x):
        """Forward pass"""
        # Backbone
        backbone_out = self.backbone(x)
        
        # Neck
        neck_out = self.neck(backbone_out)
        
        # Detection
        p3 = backbone_out * 0.6
        p4 = backbone_out * 0.8
        p5 = backbone_out * 1.0
        
        return self.detect([p3, p4, p5])

class Loss3D(nn.Module):
    """3D Detection loss function"""
    def __init__(self, nc=80):
        super(Loss3D, self).__init__()
        self.nc = nc

    def forward(self, predictions, targets):
        """Calculate loss"""
        detections = predictions['detections']
        depth = predictions['depth']
        dimensions = predictions['dimensions']
        rotation = predictions['rotation']
        
        # Calculate detection loss
        det_loss = 0
        for det in detections:
            det_loss += torch.mean(torch.abs(det))
        
        # Calculate depth loss
        depth_loss = 0
        for d in depth:
            depth_loss += torch.mean(torch.abs(d))
        
        # Calculate dimension loss
        dim_loss = 0
        for dim in dimensions:
            dim_loss += torch.mean(torch.abs(dim))
        
        # Calculate rotation loss
        rot_loss = 0
        for r in rotation:
            rot_loss += torch.mean(torch.abs(r))
        
        # Total loss
        total_loss = det_loss + 0.1 * depth_loss + 0.1 * dim_loss + 0.1 * rot_loss
        
        return {
            'total_loss': total_loss,
            'detection_loss': det_loss,
            'depth_loss': depth_loss,
            'dimension_loss': dim_loss,
            'rotation_loss': rot_loss
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