SEQUENCE_LENGTH = 12

# log calc:
# data = normalize(data), with BOUNDARIES
# data = log(data + epsilon)
# data = normalize(data), with LOG_BOUNDARIES

EPSILON = 0.1

BOUNDARIES = [
    [888.3, 868.8],
    [24.2, -8.1],
    [35.4, -0.9],
    [18.3, -25.2],
    [29.0, -10.8],
    [64.1, 3.7],
    [50.3, -19.1],
    [4.6, 1.3],
    [7.6, 2.6],
    [303.8, 57.7],
    [0.9, 0.3],
    [251.8, 0.0],
    [183.6, 1.8],
    # [183.6, 0.0]
    [200.0, 0.0]
]

DO_LOG = [
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    True,
    True,
    True
]

LOG_BOUNDARIES = [6, 0]