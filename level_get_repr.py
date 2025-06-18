import timm
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms as T
import lovely_tensors
lovely_tensors.monkey_patch()
import gc
import re
from pathlib import Path
from PIL import Image

from utils.misc import convert_mae_to_vit, convert_maskfeat_to_vit

torch.set_grad_enabled(False)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

repr_dic = "/shared/results/common/kargin/unreal_engine/features/background_complexity"
model_dic = "/shared/results/common/kargin/unreal_engine/models"
dataset_dic = "/shared/results/common/kargin/unreal_engine/dataset/background_complexity"

class HierarchicalDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = list(self.root.rglob("*.jpg"))  
        # self.samples = [p for p in self.samples if ".ipynb_checkpoints" not in str(p)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        try:
            object_class = path.parent.name
            level = path.parent.parent.name
            orientation = path.parent.parent.parent.name.replace("Frames_", "")
        except Exception as e:
            raise RuntimeError(f"Unexpected folder structure for {path}") from e

        match = re.search(r"(\d+)\.jpg$", path.name)
        file_number = int(match.group(1)) if match else -1

        metadata = {
            "path": str(path),
            "orientation": orientation,
            "level": level,
            "object_class": object_class,
            "number": file_number
        }

        return image, metadata

dataset = HierarchicalDataset(
    f"{dataset_dic}/",
    transform=T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
)
    
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)


model_names=["clip", "croco", "deit", "dino", "mae", "maskfeat", "spa"]

for model_name in model_names:
    torch.cuda.empty_cache()
    gc.collect()
    if model_name == 'dino':
        model = timm.create_model(f'vit_base_patch16_224.{model_name}', pretrained=True).cuda()
        model.head = torch.nn.Identity()
        model.fc_norm = torch.nn.Identity()
    elif model_name == "deit":
        model = timm.create_model('deit3_base_patch16_224.fb_in1k', pretrained=True).cuda()
        model.head = torch.nn.Identity()
        model.fc_norm = torch.nn.Identity()
    elif model_name == "clip":
        model = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True).cuda()
        model.head = torch.nn.Identity()
        model.fc_norm = torch.nn.Identity()
    elif model_name == "croco":
        state_dict = torch.load(f'{model_dic}/CroCo.pth', 'cpu')
        state_dict = state_dict["model"]
        state_dict = convert_mae_to_vit(state_dict)
        model = timm.create_model('vit_base_patch16_224', pretrained=False).cuda()
        msg = model.load_state_dict(state_dict, strict=False)
        use_gpu = torch.cuda.is_available() and torch.cuda.device_count()>0
        device = torch.device('cuda:0' if use_gpu else 'cpu')
        model = model.eval()
        model = model.to(device=device)
        model.head = torch.nn.Identity()
        model.fc_norm = torch.nn.Identity()
        print(msg)
    elif model_name == "mae":
        state_dict = torch.load(f'{model_dic}/mae_pretrain_vit_base.pth', 'cpu')
        state_dict = state_dict["model"]
        model = timm.create_model('vit_base_patch16_224', pretrained=False).cuda()# msg = model.load_state_dict(state_dict, strict=False)
        msg = model.load_state_dict(state_dict, strict=False)
        use_gpu = torch.cuda.is_available() and torch.cuda.device_count()>0
        device = torch.device('cuda:0' if use_gpu else 'cpu')
        model = model.eval()
        model = model.to(device=device)
        model.head = torch.nn.Identity()
        model.fc_norm = torch.nn.Identity()
        print(msg)
    elif model_name == "maskfeat":
        state_dict = torch.load(f'{model_dic}/in1k_VIT_B_MaskFeat_PT_epoch_01600.pyth', 'cpu')
        state_dict = state_dict['model_state']
        state_dict = convert_maskfeat_to_vit(state_dict)
        model = timm.create_model('vit_base_patch16_224', pretrained=False).cuda()
        msg = model.load_state_dict(state_dict, strict=False)
        use_gpu = torch.cuda.is_available() and torch.cuda.device_count()>0
        device = torch.device('cuda:0' if use_gpu else 'cpu')
        model = model.eval()
        model = model.to(device=device)
        model.head = torch.nn.Identity()
        model.fc_norm = torch.nn.Identity()
        print(msg)
    elif model_name == "spa":
        state_dict = torch.load(f"{model_dic}/spa-b.ckpt")
        state_dict = state_dict["state_dict"]

        for k in list(state_dict.keys()):
            if not k.startswith("model.img_backbone"):
                del state_dict[k]
            else:
                new_key = k.replace("model.img_backbone.", "")
                state_dict[new_key] = state_dict.pop(k)

        model = timm.create_model('vit_base_patch16_224', pretrained=False).cuda()
        model.head = torch.nn.Identity()
        model.fc_norm = torch.nn.Identity()

        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Model {model_name} not found")

    print(model_name)
    result = {
        "features": [],
        "orientation": [],
        "level": [],
        "object_class": [],
        "number": [],
        "path": [],
    }

    model.eval()
    for images, metadata in tqdm(dataloader):
        images = images.to(device)
        with torch.no_grad():
            feats = model.forward_features(images)

        result["features"].extend(feats.cpu())
        for key in metadata:
            result[key].extend(metadata[key])

    result["features"] = torch.stack(result["features"]).half()

    # Clear GPU and unused RAM
    del model
    torch.cuda.empty_cache()
    gc.collect()

    torch.save(result, f"{repr_dic}/repr_{model_name}.pt")