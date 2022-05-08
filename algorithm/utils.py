def to_transform(vocab, targets, unknown_value):
    sources = []
    for t in targets:
        if t in vocab:
            s = vocab[t]
        else:
            s = unknown_value
        sources.append(s)
    return sources


def laplace_smoothing(x, y, c, smooth=1e-6):
    return (x + smooth) / (y + c * smooth)
