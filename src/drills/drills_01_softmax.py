import numpy as np


def softmax(x, axis=-1):
    """Compute the softmax of a numpy array along the given axis.

    Args:
        x: Input array.
        axis: Axis along which the softmax is computed.

    Returns:
        Array of the same shape as x, with values normalized along axis.
    """
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis, keepdims=True)


def log_softmax(x, axis=-1):
    """Compute the log-softmax of a numpy array along the given axis.

    Args:
        x: Input array.
        axis: Axis along which the log-softmax is computed.

    Returns:
        Array of the same shape as x, with log-probabilities along axis.
    """
    x = x - np.max(x)
    log_sum_x = np.log(np.sum(np.exp(x), axis, keepdims=True))
    return x - log_sum_x


def cross_entropy(logits, targets):
    """Compute the mean cross entropy loss between logits and target classes.

    Args:
        logits: Array of shape (N, C) with unnormalized class scores.
        targets: Array of shape (N,) with the ground-truth class index for
            each of the N examples.

    Returns:
        The mean cross entropy loss across the N examples, as a float.
    """
    log_probs = log_softmax(logits, axis=-1)  # log-probabilities for every class, per example
    num_samples = logits.shape[0]  # N, the batch size
    picked = log_probs[np.arange(num_samples), targets]  # log P(true class) for each example
    return float(-picked.mean())  # average negative log-likelihood across the batch
