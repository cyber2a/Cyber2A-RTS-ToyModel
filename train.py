import torch

from toy_model.dataset import RTSDataset
from toy_model.engine import evaluate, train_one_epoch
from toy_model.model import get_model_instance_segmentation
from toy_model.transforms import get_transform
from toy_model.utils import collate_fn


def main():
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load datasets
    dataset = RTSDataset("data/coco_rts_train.json", get_transform(train=True))
    dataset_test = RTSDataset("data/coco_rts_valtest.json", get_transform(train=False))

    # Create data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # Initialize model
    model = get_model_instance_segmentation(num_classes=2)
    model.to(device)

    # Set up optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

    # Save the model
    torch.save(model.state_dict(), "rts_model.pth")


if __name__ == "__main__":
    main()
