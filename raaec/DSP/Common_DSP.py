def wav_norm(x):
    x_max, x_min = x.max(), x.min()
    a = (x_max + x_min) / 2
    b = (x_max - x_min) / 2
    return (x - a) / b