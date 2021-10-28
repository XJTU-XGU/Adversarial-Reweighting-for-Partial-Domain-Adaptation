def recommended_bottleneck_dim(num_class):
    j = 8
    while True:
        if 3*num_class <= 256:
            dim = 256
            break
        elif 3*num_class > 2**j and 3*num_class <= 2**(j+1):
            dim = 2**(j+1)
            break
        j += 1
    return dim
