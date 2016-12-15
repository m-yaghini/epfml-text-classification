def linecount(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines


def give_vector(word, K):
    import numpy as np
    print(word.shape)
    vec = np.array(word).reshape(1, 1)
    return vec
