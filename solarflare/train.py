import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SolarFlareDataset, transform
from model import SolarCNN
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv',   default='solarflaredata/Train_Data_by_AR_png_224.csv')
    parser.add_argument('--val_csv',     default='solarflaredata/Validation_Data_by_AR_png_224.csv')
    parser.add_argument('--images_dir',  default='solarflaredata/Lat60_Lon60_Nans0_png_224/')
    parser.add_argument('--epochs',      type=int, default=10)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    training_dataset = SolarFlareDataset(args.train_csv, args.images_dir, transform=transform)
    validation_dataset = SolarFlareDataset(args.val_csv, args.images_dir, transform=transform)

    train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SolarCNN().to(device)
    positive_weight = torch.tensor([31.2]).to(device) #had to calculate when we get a class 0 vs class 1 solar flare
    criteria = nn.BCEWithLogitsLoss(pos_weight = positive_weight) #Calculating our BCE loss via the rarity of events
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.01)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for image, labels in progress:
            image = image.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(image).squeeze(1)
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f'{loss.item():.4f}')

        average_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/10 complete — Avg Loss: {average_loss:.4f}')
        torch.save(model.state_dict(), f'checkpoint_epoch{epoch+1}.pth')
