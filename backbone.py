import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoModel, AutoImageProcessor


def get_processor(backbone_name):
    backbone_name = backbone_name.lower()
    if backbone_name == 'dinov2':
        model_name = 'facebook/dinov2-base'
        processor = AutoImageProcessor.from_pretrained(model_name)
    elif backbone_name == 'mocov3':
        processor = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone_name}")
    
    return processor

class Adapter(nn.Module):
    def __init__(self, hidden_dim, rank, alpha=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank  
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.enabled = True

        nn.init.kaiming_uniform_(self.down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        if self.enabled:
            return self.up(self.down(x)) * (self.alpha / self.rank)
        else:
            return torch.zeros_like(x)


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, alpha=None):
        super().__init__()
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank

        self.lora_down = nn.Linear(self.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, self.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        out = self.linear(x)
        lora_out = self.lora_up(self.lora_down(x)) * (self.alpha / self.rank)
        return out + lora_out


def get_parent_module(model, name):
    components = name.split('.')
    for comp in components[:-1]:
        model = getattr(model, comp)
    return model


def inject_lora_qkv(model, rank, alpha=None):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'qkv' in name:
            parent = get_parent_module(model, name)
            key = name.split('.')[-1]
            original = getattr(parent, key)
            setattr(parent, key, LoRALinear(original, rank=rank, alpha=alpha))


def inject_lora_dino(model, rank, alpha=None):
    for layer in model.encoder.layer:
        sa = layer.attention.attention
        sa.query = LoRALinear(sa.query, rank=rank, alpha=alpha)
        sa.value = LoRALinear(sa.value, rank=rank, alpha=alpha)


def inject_adapters(backbone_name, model, rank, alpha=None):
    blocks = model.encoder.layer if backbone_name == 'dinov2' else model.blocks
    for block in blocks:
        mlp = block.mlp
        adapter = Adapter(mlp.fc2.out_features, rank, alpha)
        block.adapter = adapter

        forward = mlp.forward
        def adapter_forward(x, forward=forward, adapter=adapter):
            return forward(x) + adapter(x)

        mlp.forward = adapter_forward


def map_labels(labels):
    unique = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique)}
    mapped = [label_map[label] for label in labels]
    return torch.tensor(mapped), label_map


class Backbone:
    def __init__(self, args):
        self.args = args
        self.backbone_name = args.backbone.lower()
        self.device = args.device  

        if self.backbone_name == 'dinov2':
            from transformers import AutoImageProcessor, AutoModel
            model_name = 'facebook/dinov2-base'
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.num_features=768
        elif self.backbone_name == 'mocov3':
            # self.model = timm.create_model('vit_base_patch16_224', pretrained=False)
            # checkpoint = torch.load('vit-b-300ep.pth.tar', map_location='cpu')
            # state_dict = checkpoint['state_dict']
            # state_dict = {k.replace('module.base_encoder.', ''): v for k, v in state_dict.items()}
            # self.model.load_state_dict(state_dict, strict=False)
            # self.model = self.model.to(self.device)
            # self.num_features=768
            
            model = timm.create_model('vit_base_patch16_224', pretrained=False)
            ckpt = torch.load('mocov3-vit-base-300ep.pth', map_location='cpu')['model']
            state_dict = model.state_dict()
            state_dict.update(ckpt)
            model.load_state_dict(state_dict)
            self.model = model.to(self.device)
            self.num_features=768
        else:
            raise ValueError(f"Unknown backbone name: {self.backbone_name}")

    def forward(self, inputs, train=False, concat_features=False):
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            if self.backbone_name == 'dinov2':
                pixel_values = inputs['pixel_values'][0].to(self.device)
                features = self.model(pixel_values).last_hidden_state[:, 0]
            elif self.backbone_name == 'mocov3':
                inputs = inputs.to(self.device)
                features = self.model.forward_features(inputs)[:, 0]

            if concat_features:
                if not hasattr(self, 'adapters'):
                    raise RuntimeError("Adapters are not defined. Cannot concat features without adapters.")
                for adapter in self.adapters: adapter.enabled = False
                if self.backbone_name == 'dinov2':
                    pixel_values = inputs['pixel_values'][0].to(self.device)
                    no_adapter = self.model(pixel_values).last_hidden_state[:, 0]
                elif self.backbone_name == 'mocov3':
                    inputs = inputs.to(self.device)
                    no_adapter = self.model.forward_features(inputs)[:, 0]

                for adapter in self.adapters: adapter.enabled = True
                features = torch.cat([features, no_adapter], dim=-1)
            return features

    def get_features(self, dataloader, concat_features=False):
        all_features, all_labels = [], []
        for batch in tqdm(dataloader):
            images, labels = batch
            features = self.forward(images, concat_features=concat_features).cpu().numpy()
            all_features.append(features)
            all_labels.append(labels.numpy())
        return np.concatenate(all_features), np.concatenate(all_labels)

    def finetune(self, dataloader):
        args = self.args
        all_labels = [label.item() for _, labels in dataloader for label in labels]
        _, label_map = map_labels(all_labels)

        num_classes = len(label_map)
        classifier = nn.Linear(self.num_features, num_classes).to(self.device)

        param_groups = []
        method = args.training_method.lower()
        if method == 'aper':
            inject_adapters(self.backbone_name, self.model, args.rank)
            self.model.to(self.device)
            
            for p in self.model.parameters():
                p.requires_grad = False  
            self.adapters = [block.adapter for block in (self.model.encoder.layer if self.backbone_name == 'dinov2' else self.model.blocks)]
            adaptor_params = [param for adapter in self.adapters for param in adapter.parameters()]
            for param in adaptor_params: param.requires_grad = True
            param_groups = [
                {"params": classifier.parameters(), "lr": args.learning_rate},
                {"params": adaptor_params, "lr": args.lora_learning_rate}
            ]
        
        elif method == 'slca++':
            if self.backbone_name == 'dinov2':
                inject_lora_dino(self.model, rank=args.rank, alpha=args.rank)
            elif self.backbone_name == 'mocov3':
                inject_lora_qkv(self.model, rank=args.rank, alpha=args.rank)
            self.model.to(self.device)
            
            for p in self.model.parameters():
                p.requires_grad = False
            lora_params = [p for n, p in self.model.named_parameters() if 'lora_' in n]
            for param in lora_params: param.requires_grad = True
            param_groups = [
                {"params": classifier.parameters(), "lr": args.learning_rate},
                {"params": lora_params, "lr": args.lora_learning_rate}
            ]
            
        elif method == 'slca':
            for p in self.model.parameters():
                p.requires_grad = True
            param_groups = [
                {"params": classifier.parameters(), "lr": args.learning_rate},
                {"params": self.model.parameters(), "lr": args.backbone_learning_rate}
            ]
        else:
            raise ValueError(f"Unknown training method name: {method}")

        optimizer = torch.optim.Adam(param_groups)

        self.model.train()
        classifier.train()
        for epoch in range(args.num_epochs):
            total_loss = 0
            for images, labels in dataloader:
                labels = torch.tensor([label_map[l.item()] for l in labels]).to(self.device)
                features = self.forward(images, train=True)
                logits = classifier(features)
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {total_loss:.4f}")


class BackboneLORA(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, rank: int, device: torch.device):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.device = device

        if self.backbone_name == 'dinov2':
            model_name = 'facebook/dinov2-base'
            model = AutoModel.from_pretrained(model_name)
            num_features = model.config.hidden_size

            for param in model.parameters():
                param.requires_grad = False
            
            inject_lora_dino(model, rank=rank)

        elif self.backbone_name == 'mocov3':
            model = timm.create_model('vit_base_patch16_224', pretrained=False)
            ckpt = torch.load('mocov3-vit-base-300ep.pth', map_location='cpu')['model']
            state_dict = model.state_dict()
            state_dict.update(ckpt)
            model.load_state_dict(state_dict)
            num_features=768

            for param in model.parameters():
                param.requires_grad = False

            inject_lora_qkv(model, rank=rank)
        else:
            raise ValueError(f"Unknown backbone name: {backbone_name}")

        # Enable LoRA parameters
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                for p in module.lora_down.parameters():
                    p.requires_grad = True
                for p in module.lora_up.parameters():
                    p.requires_grad = True

        self.model = model.to(device)
        self.classifier = nn.Linear(num_features, num_classes).to(device)

    def forward(self, inputs):
        if self.backbone_name == 'dinov2':
            pixel_values = inputs['pixel_values'][0].to(self.device)
            features = self.model(pixel_values).last_hidden_state[:, 0]  # CLS token
        elif self.backbone_name == 'mocov3':
            inputs = inputs.to(self.device)
            features = self.model.forward_features(inputs)[:, 0]
        else:
            raise ValueError("Unsupported backbone")

        return self.classifier(features)

