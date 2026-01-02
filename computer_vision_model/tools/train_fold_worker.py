import os
import sys
sys.path.append("")

def train_fold_loop(fold, config, shared_dict, barrier, save_flag):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(fold)
    os.environ['TORCH_HOME'] = ''

    import torch
    from torch.utils.data import DataLoader
    from models.resnet import build_resnet50
    import pandas as pd
    from data.dataset import CropDataset
    # Torch
    device = torch.device("cuda:0")  # Always cuda:0 after masking
    torch.cuda.set_device(device)

    # Data
    country=config['country']
    year=config['year']
    csv_path=os.path.join(config['paths']['dataset_dir'], f"{country}_{year}.csv")
    df = pd.read_csv(csv_path)

    # Create consistent zone2idx mapping
    zones = sorted(df['zone'].unique())
    zone2idx = {zone: i for i, zone in enumerate(zones)}
    train_transform = CropDataset.build_transform(True, 224)
    val_transform = CropDataset.build_transform(False, 224)
    train_dataset = CropDataset(csv_path=csv_path, folds=[i for i in range(1, 6) if i != fold], zone2idx=zone2idx, config=config, transform=train_transform)
    val_dataset = CropDataset(csv_path=csv_path, folds=[fold], zone2idx=zone2idx, config=config, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['val']['batch_size'], shuffle=False, num_workers=4)

    model = build_resnet50()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = torch.nn.MSELoss()

    # Train Loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        for images, labels, zone_ids in train_loader:
            images = images.to(device)
            zone_ids = zone_ids.to(device)
            labels = labels.unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        val_preds, val_targets, val_zone_ids = validate(model, val_loader, device)
        shared_dict[fold] = {
            "preds": val_preds,
            "targets": val_targets,
            "zones": val_zone_ids
        }
        barrier.wait()
        barrier.wait()
        if save_flag.value == 1:
            torch.save(
                model.state_dict(),
                os.path.join(config['paths']['ckpt_dir'],f"{country}_{year}/fold_{fold}_epoch_{epoch}.pt")
            )
        barrier.wait()
        barrier.wait()

def validate(model, val_loader, device):
    import torch
    model.eval() 
    all_preds = []
    all_targets = []
    all_zone_ids=[]
    with torch.no_grad():
        for images, labels, zone_ids in val_loader:
            images = images.to(device)
            zone_ids = zone_ids.to(device)
            labels = labels.unsqueeze(1).to(device)
            outputs=model(images)
            all_preds.append(outputs.cpu())
            all_targets.append(labels.cpu())
            all_zone_ids.append(zone_ids.cpu())
    all_preds = torch.cat(all_preds).numpy().flatten()
    all_targets = torch.cat(all_targets).numpy().flatten()
    all_zone_ids = torch.cat(all_zone_ids).numpy()
    return all_preds, all_targets, all_zone_ids
