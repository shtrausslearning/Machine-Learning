from torch.utils.data import DataLoader

# Training & Validation dataloaders
# train_ts -  val_dl - 

train_dl = DataLoader(train_ts,
                      batch_size=32,
                      shuffle=True)
val_dl = DataLoader(val_ts,
                    batch_size=32,
                    shuffle=False)

for x,y in train_dl:
    print(x.shape,y)
    break
