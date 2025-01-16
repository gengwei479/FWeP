import math

def remap_curve_00(input):
    # input range: [-pi, pi]
    # output range: [-1, 1]
    border = math.pi / 2
    input = (input >= border) * border + (input < border and input > -border) * input + (input <= -border) * (-border)
    output = math.sin(input)
    return output

def remap_curve_01(input, param = 8):
    return math.atan(param * input) * 2 / math.pi

def remap_curve_02(input, param = 0.01):
    border = math.pi / (4 * param)
    input = (input >= border) * border + (input < border and input > -border) * input + (input <= -border) * (-border)
    output = math.tan(param * input)
    return -output