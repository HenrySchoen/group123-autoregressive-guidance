"""Load the frozen train/val/test splits and show what a batch looks like."""
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset


def get_loaders(batch_size=32):
    train_data = ProteinDataset("data/train.csv")
    val_data = ProteinDataset("data/val.csv")
    test_data = ProteinDataset("data/test.csv")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_data, val_data, test_data, train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_data, val_data, test_data, train_loader, val_loader, test_loader = get_loaders()
    print(train_data)
    print(val_data)
    print(test_data)

    x, x_shifted, y, length = next(iter(train_loader))
    print("x:             ", tuple(x.shape), x.dtype)
    print("x_shifted:     ", tuple(x_shifted.shape), x_shifted.dtype)
    print("y:             ", tuple(y.shape), y[:8].tolist())
    print("length:        ", tuple(length.shape), length[:8].tolist())
    print("first sequence:", train_data.decode(x[0]))
