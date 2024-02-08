import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def compute_val_loss(loader, model, device):
    """Compute the validation loss"""
    model.eval()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        test_loss = F.nll_loss(out, data.y)
        total_loss += float(test_loss) * data.num_graphs
    return total_loss / len(loader.dataset)


def train(train_loader, model, optimizer, device):
    """Train the model for a single epoch and return the train loss"""
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader, model, device):
    """Compute the accuracy on the given dataset."""
    model.eval()

    total_correct = total_examples = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
        total_examples += data.num_graphs

    return total_correct / total_examples


def plots(
    loss_train_history,
    loss_valid_history,
    acc__valid_history,
    acc__test__history,
    model_desc,
):
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    axs[0].plot(loss_train_history, label="train")
    axs[0].plot(loss_valid_history, label="val")
    axs[0].legend()
    axs[0].set_title("loss")
    axs[0].set_xlabel("epochs")
    axs[0].grid()
    # nbins = int(((len(loss_train_history) + 1 ) / 10)*2 + 2 )
    # axs[0].locator_params(axis='both', nbins=nbins)

    axs[1].plot(acc__valid_history, label="val_accuracy")
    axs[1].plot(acc__test__history, label="test_accuracy")
    axs[1].set_title("accuracy")
    axs[1].set_xlabel("epochs")
    axs[1].legend()
    axs[1].grid()

    fig.suptitle(model_desc, fontsize=10)
