# deepgym

- learn models from data
- reuse same data as model-free agents
- see https://www.zotero.org/rrsc/collections/WGHWZ33I

#### pytorch Dataset
```
class ...(Dataset):
   ...
    def __getitem__(self, idx):
        ...
        return {'image': transforms.ToTensor()(image),
                'labels': label
                }

data_loader = torch.utils.data.DataLoader(dataset, ...

```
