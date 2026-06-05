import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SolarFlareDataset, transform
from model import SolarCNN
from tqdm import tqdm



device = torch.device('mps')

#labels given to me by Claude so I don't have to repeat the links
TRAIN_CSV   = '/Users/diegogonzalez/satellitemapping/solarflare/solarflaredata/Train_Data_by_AR_png_224.csv'
VAL_CSV     = '/Users/diegogonzalez/satellitemapping/solarflare/solarflaredata/Validation_Data_by_AR_png_224.csv'
IMAGES_DIR  = '/Users/diegogonzalez/satellitemapping/solarflare/solarflaredata/Lat60_Lon60_Nans0_png_224/'  # folder after extraction

if __name__ == '__main__':
    training_dataset = SolarFlareDataset(TRAIN_CSV, IMAGES_DIR, transform = transform)
    validation_dataset = SolarFlareDataset(VAL_CSV, IMAGES_DIR, transform = transform)

    train_loader = DataLoader(training_dataset, batch_size = 32, shuffle = True, num_workers = 0)
    validation_loader = DataLoader(validation_dataset, batch_size = 32, shuffle = False, num_workers = 0)

    model = SolarCNN().to(device)
    positive_weight = torch.tensor([31.2]).to(device) #had to calculate when we get a class 0 vs class 1 solar flare
    criteria = nn.BCEWithLogitsLoss(pos_weight = positive_weight) #Calculating our BCE loss via the rarity of events
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.01)

    for epoch in range(10):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/10')
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
