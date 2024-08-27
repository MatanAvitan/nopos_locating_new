import tqdm
import numpy as np
import torch

def loss_fn(logits, labels, per_token=False):
    # logit shape: [batch, pos, vocab]
    # token shape: [batch, pos]
    log_probs = logits.log_softmax(-1)
    correct_log_probs = log_probs.gather(-1, labels[..., None])[..., 0] # The gathering operation occurs on the logit dimension
    if per_token:
        return -correct_log_probs
    else:
        return -correct_log_probs.mean()

def evaluate(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for tokens, targets in tqdm.tqdm(test_loader, desc="Evaluating"):
            tokens=tokens.to(device)
            targets=targets.to(device)
            # Forward pass
            logits = model(tokens)
            # Calculate loss
            loss = loss_fn(logits, targets)
            total_loss += loss.item()

            # Convert logits to predicted labels (assuming you are using softmax)
            predicted_labels = logits.argmax(dim=-1)
            # Append targets and predictions for accuracy calculation
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())

    # Calculate accuracy
    all_predictions_arr, all_targets_arr = np.array(all_predictions), np.array(all_targets)
    accuracy = (all_predictions_arr == all_targets_arr).sum().item() / all_predictions_arr.size

    # Calculate average loss
    average_loss = total_loss / len(test_loader)

    return accuracy, average_loss
# evaluate(model, test_loader, device)
