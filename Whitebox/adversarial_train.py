import argparse
import copy
import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils


NUM_CLASSES = 43


def build_train_val_loaders(data_dir, batch_size, num_workers, seed, max_train_samples=None):
    train_csv = os.path.join(data_dir, "Train.csv")
    full_df = pd.read_csv(train_csv).reset_index(drop=True)

    shuffled = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_train_samples is not None:
        shuffled = shuffled.iloc[:max_train_samples].reset_index(drop=True)

    val_size = max(1, int(len(shuffled) * 0.1))
    val_df = shuffled.iloc[:val_size].reset_index(drop=True)
    train_df = shuffled.iloc[val_size:].reset_index(drop=True)

    train_split_csv = os.path.join("/tmp", "train_split.csv")
    val_split_csv = os.path.join("/tmp", "val_split.csv")
    train_df.to_csv(train_split_csv, index=False)
    val_df.to_csv(val_split_csv, index=False)

    train_dataset = utils.GTSRBKaggleDataset(train_split_csv, data_dir)
    val_dataset = utils.GTSRBKaggleDataset(val_split_csv, data_dir)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def load_target_model(checkpoint_path, device, full_model):
    if checkpoint_path is None:
        return utils.ResNet18_L3(num_classes=NUM_CLASSES).to(device)

    if full_model:
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return model.to(device)

    model = utils.ResNet18_L3(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def train_one_epoch(model, loader, criterion, device, optimizer, epsilon, alpha, num_steps):
    model.train()

    total_adv_loss = 0.0
    total_adv_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        adv_images = utils.generate_adversarial_batch(
            model=model,
            images=images,
            labels=labels,
            attack="PGD",
            device=device,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            criterion=criterion,
        )

        optimizer.zero_grad(set_to_none=True)
        clean_outputs = model(utils.normalize_batch(images))
        clean_loss = criterion(clean_outputs, labels)
        adv_outputs = model(utils.normalize_batch(adv_images))
        adv_loss = criterion(adv_outputs, labels)
        loss = 0.5 * clean_loss + 0.5 * adv_loss
        loss.backward()
        optimizer.step()

        total_adv_loss += adv_loss.item() * labels.size(0)
        total_adv_correct += (adv_outputs.argmax(dim=1) == labels).sum().item()
        total_seen += labels.size(0)

    return total_adv_loss / total_seen, total_adv_correct / total_seen


def evaluate_clean_and_adv(model, loader, criterion, device, epsilon, alpha, num_steps):
    model.eval()

    clean_loss_sum = 0.0
    adv_loss_sum = 0.0
    clean_correct = 0
    adv_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        adv_images = utils.generate_adversarial_batch(
            model=model,
            images=images,
            labels=labels,
            attack="PGD",
            device=device,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            criterion=criterion,
        )

        with torch.no_grad():
            clean_outputs = model(utils.normalize_batch(images))
            adv_outputs = model(utils.normalize_batch(adv_images))
            clean_loss = criterion(clean_outputs, labels)
            adv_loss = criterion(adv_outputs, labels)

        clean_loss_sum += clean_loss.item() * labels.size(0)
        adv_loss_sum += adv_loss.item() * labels.size(0)
        clean_correct += (clean_outputs.argmax(dim=1) == labels).sum().item()
        adv_correct += (adv_outputs.argmax(dim=1) == labels).sum().item()
        total_seen += labels.size(0)

    return {
        "clean_loss": clean_loss_sum / total_seen,
        "adv_loss": adv_loss_sum / total_seen,
        "clean_acc": clean_correct / total_seen,
        "adv_acc": adv_correct / total_seen,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--full_model", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=8 / 255.0)
    parser.add_argument("--alpha", type=float, default=2 / 255.0)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(args.data_dir or os.path.join(base_dir, "..", "GTSRB"))
    save_dir = os.path.normpath(args.save_dir or os.path.join(base_dir, "results"))
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    init_checkpoint = args.init_checkpoint
    if init_checkpoint is not None and not os.path.isabs(init_checkpoint):
        init_checkpoint = os.path.normpath(os.path.join(base_dir, "..", init_checkpoint))

    train_loader, val_loader = build_train_val_loaders(
        data_dir, args.batch_size, args.num_workers, args.seed, args.max_train_samples
    )

    model = load_target_model(init_checkpoint, device, args.full_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_path = os.path.join(save_dir, "resnet18_section52_adv_trained.pth")

    best_val_adv_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    print(f"Device            : {device}")
    print(f"Data dir          : {data_dir}")
    print(f"Init checkpoint   : {init_checkpoint}")
    print(f"Epsilon           : {args.epsilon:.6f}")
    print(f"Alpha             : {args.alpha:.6f}")
    print(f"PGD steps         : {args.num_steps}")
    print(f"Train/Val samples : {len(train_loader.dataset)}/{len(val_loader.dataset)}")

    for epoch in range(args.epochs):
        train_adv_loss, train_adv_acc = train_one_epoch(
            model, train_loader, criterion, device, optimizer,
            args.epsilon, args.alpha, args.num_steps
        )
        val_metrics = evaluate_clean_and_adv(
            model, val_loader, criterion, device,
            args.epsilon, args.alpha, args.num_steps
        )
        print(
            f"[Epoch {epoch + 1:02d}/{args.epochs}] "
            f"Train Adv Loss: {train_adv_loss:.4f}, Train Adv Acc: {train_adv_acc:.4f} | "
            f"Val Clean Acc: {val_metrics['clean_acc']:.4f}, "
            f"Val Adv Acc: {val_metrics['adv_acc']:.4f}"
        )

        if val_metrics["adv_acc"] > best_val_adv_acc:
            best_val_adv_acc = val_metrics["adv_acc"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
            print(f"Saved checkpoint  : {checkpoint_path}")

    model.load_state_dict(best_state)
    final_metrics = evaluate_clean_and_adv(
        model, val_loader, criterion, device,
        args.epsilon, args.alpha, args.num_steps
    )
    print(f"Best Val Clean Acc: {final_metrics['clean_acc']:.4f}")
    print(f"Best Val Adv Acc  : {final_metrics['adv_acc']:.4f}")


if __name__ == "__main__":
    main()
