def get_discrete_bin(x):
    if x <= -3:
        return 0
    elif x <= -2:
        return 1
    elif x <= -1.5:
        return 2
    elif x <= -1:
        return 3
    elif x <= -0.8:
        return 4
    elif x <= -0.6:
        return 5
    elif x <= -0.4:
        return 6
    elif x <= -0.2:
        return 7
    elif x <= 0:
        return 8
    elif x < 0.2:
        return 9
    elif x < 0.4:
        return 10
    elif x < 0.6:
        return 11
    elif x < 0.8:
        return 12
    elif x < 1:
        return 13
    elif x < 1.5:
        return 14
    elif x < 2:
        return 15
    elif x < 3:
        return 16
    else:
        return 17