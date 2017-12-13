import numpy as np
from sympy.tensor.array import derive_by_array

# Idee:
#theoretically for given epsilon there is a delta, such that each point in a delta neighborhood around x_0 has error less than epsilon to tangent cone
#approximate variety linearly by using PCA (i.e. reduce dimension by directions corresponding to vanishing eigenvalues of scatter matrix of sampling)
#Difficulties: Identify singular points. Solve this by exhaustively check for isolated zeros (in case for intersections of variety)


#Numerical Parameters
precision = 10**(-20) # set numerical precision
epsilon_eig_val = 10**(-17) # set cut of eigenvalues
espilon_lin_approx = 10**(-15) # set epsilon for error of tangent cone

#Structure Parameters
delta = 1 # prescribed periodicity of structure


def f(x):
    f= [0,
         (-x[0] + 0.9) ** 2 + (-x[1] + 0.5) ** 2 + (-x[2] - x[66] + 0.75) ** 2 - 0.1075,
         (-x[5] + 0.9) ** 2 + (-x[6] + 0.5) ** 2 + (-x[7] + 0.75) ** 2 - 0.1075,
         (-x[26] + 0.35) ** 2 + (-x[27] + 0.5) ** 2 + (-x[28] + 0.75) ** 2 - 0.1075,
         (-x[17] + 0.35) ** 2 + (-x[18] + 0.5) ** 2 + (-x[19] + 0.75) ** 2 - 0.1075,
         (x[41] - x[53]) ** 2 + (x[42] - x[54]) ** 2 + (x[43] - x[55]) ** 2 - 0.3025,
         (x[35] - x[41]) ** 2 + (x[36] - x[42]) ** 2 + (x[37] - x[43]) ** 2 - 0.1075,
         (x[26] - x[53]) ** 2 + (x[27] - x[54]) ** 2 + (x[28] - x[55]) ** 2 - 0.1075,
         (x[17] - x[53]) ** 2 + (x[19] - x[55]) ** 2 + (x[18] - x[54] - x[65]) ** 2 - 0.1075,
         (-x[38] + x[41]) ** 2 + (-x[39] + x[42] + x[65]) ** 2 + (-x[40] + x[43] - x[66]) ** 2 - 0.1075,
         (x[17] - x[23]) ** 2 + (x[18] - x[24]) ** 2 + (x[19] - x[25]) ** 2 - 0.3025,
         (x[10] - x[25]) ** 2 + (-x[23] + x[8]) ** 2 + (-x[24] + x[9]) ** 2 - 0.1075,
         (x[23] - x[50]) ** 2 + (x[24] - x[51]) ** 2 + (x[25] - x[52]) ** 2 - 0.1075,
         (-x[10] + x[7]) ** 2 + (x[5] - x[8]) ** 2 + (x[6] - x[9]) ** 2 - 0.3025,
         (x[10] - x[64]) ** 2 + (-x[63] + x[9]) ** 2 + (-delta - x[62] + x[8]) ** 2 - 0.1075,
         (x[3] - x[5]) ** 2 + (x[4] - x[7]) ** 2 + (-x[6] + 0.5) ** 2 - 0.1075,
         (x[0] - x[11]) ** 2 + (-x[12] + x[1]) ** 2 + (-x[13] + x[2]) ** 2 - 0.3025,
         (x[0] - x[3]) ** 2 + (x[1] - 0.5) ** 2 + (x[2] - x[4]) ** 2 - 0.1075,
         (x[11] - x[29]) ** 2 + (x[12] - x[30]) ** 2 + (x[13] - x[31]) ** 2 - 0.1075,
         (x[20] - x[62]) ** 2 + (x[21] - x[63]) ** 2 + (x[22] - x[64]) ** 2 - 0.3025,
         (x[14] - x[20]) ** 2 + (x[15] - x[21]) ** 2 + (x[16] - x[22]) ** 2 - 0.1075,
         (x[50] - x[62]) ** 2 + (x[51] - x[63]) ** 2 + (x[52] - x[64]) ** 2 - 0.1075,
         (-x[22] + x[49]) ** 2 + (-delta - x[20] + x[47]) ** 2 + (-x[21] + x[48] + x[65]) ** 2 - 0.1075,
         (x[35] - x[50]) ** 2 + (x[37] - x[52]) ** 2 + (x[36] - x[51] + x[65]) ** 2 - 0.3025,
         (x[32] - x[35]) ** 2 + (x[33] - x[36]) ** 2 + (x[34] - x[37]) ** 2 - 0.1075,
         (x[33] - x[48]) ** 2 + (x[34] - x[49]) ** 2 + (delta + x[32] - x[47]) ** 2 - 0.3025,
         (-x[48] + x[57]) ** 2 + (-x[49] + x[58]) ** 2 + (delta - x[47] + x[56]) ** 2 - 0.1075,
         (x[12] - x[60]) ** 2 + (-delta + x[11] - x[59]) ** 2 + (x[13] - x[61] + x[66]) ** 2 - 0.1075,
         (x[56] - x[59]) ** 2 + (x[57] - x[60]) ** 2 + (x[58] - x[61] + x[66]) ** 2 - 0.3025,
         (x[44] - x[59]) ** 2 + (x[45] - x[60]) ** 2 + (x[46] - x[61] + x[66]) ** 2 - 0.1075,
         (x[14] - x[56]) ** 2 + (x[15] - x[57]) ** 2 + (x[16] - x[58]) ** 2 - 0.1075,
         (-x[32] + x[38]) ** 2 + (-x[34] + x[40]) ** 2 + (-x[33] + x[39] - x[65]) ** 2 - 0.1075,
         (x[38] - x[44]) ** 2 + (x[40] - x[46]) ** 2 + (x[39] - x[45] - x[65]) ** 2 - 0.3025,
         (x[29] - x[44]) ** 2 + (x[30] - x[45]) ** 2 + (x[31] - x[46]) ** 2 - 0.1075,
         (x[26] - x[29]) ** 2 + (x[27] - x[30]) ** 2 + (x[28] - x[31] - x[66]) ** 2 - 0.3025,
         (-x[15] + 0.5) ** 2 + (-x[16] + x[4]) ** 2 + (-delta - x[14] + x[3]) ** 2 - 0.3025]

    return f

def df(x):
    df = [
        [0, 2 * x[0] - 1.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[0] - 2 * x[11], 2 * x[0] - 2 * x[3], 0, 0,
         0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2 * x[1] - 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[12] + 2 * x[1], 2 * x[1] - 1.0, 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2 * x[2] + 2 * x[66] - 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[13] + 2 * x[2],
         2 * x[2] - 2 * x[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[3] - 2 * x[5], 0, -2 * x[0] + 2 * x[3], 0, 0, 0, 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * delta - 2 * x[14] + 2 * x[3]],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[4] - 2 * x[7], 0, -2 * x[2] + 2 * x[4], 0, 0, 0, 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[16] + 2 * x[4]],
        [0, 0, 2 * x[5] - 1.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[5] - 2 * x[8], 0, -2 * x[3] + 2 * x[5], 0, 0, 0, 0,
         0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2 * x[6] - 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[6] - 2 * x[9], 0, 2 * x[6] - 1.0, 0, 0, 0, 0, 0, 0,
         0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2 * x[7] - 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[10] + 2 * x[7], 0, -2 * x[4] + 2 * x[7], 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[23] + 2 * x[8], 0, -2 * x[5] + 2 * x[8],
         -2 * delta - 2 * x[62] + 2 * x[8],
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[24] + 2 * x[9], 0, -2 * x[6] + 2 * x[9], -2 * x[63] + 2 * x[9], 0, 0,
         0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[10] - 2 * x[25], 0, 2 * x[10] - 2 * x[7], 2 * x[10] - 2 * x[64], 0, 0,
         0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[0] + 2 * x[11], 0, 2 * x[11] - 2 * x[29], 0, 0, 0, 0, 0,
         0,
         0, 0, -2 * delta + 2 * x[11] - 2 * x[59], 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[12] - 2 * x[1], 0, 2 * x[12] - 2 * x[30], 0, 0, 0, 0, 0,
         0,
         0, 0, 2 * x[12] - 2 * x[60], 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[13] - 2 * x[2], 0, 2 * x[13] - 2 * x[31], 0, 0, 0, 0, 0,
         0,
         0, 0, 2 * x[13] - 2 * x[61] + 2 * x[66], 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[14] - 2 * x[20], 0, 0, 0, 0, 0, 0, 0, 0, 0,
         2 * x[14] - 2 * x[56], 0, 0, 0, 0, 2 * delta + 2 * x[14] - 2 * x[3]],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[15] - 2 * x[21], 0, 0, 0, 0, 0, 0, 0, 0, 0,
         2 * x[15] - 2 * x[57], 0, 0, 0, 0, 2 * x[15] - 1.0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[16] - 2 * x[22], 0, 0, 0, 0, 0, 0, 0, 0, 0,
         2 * x[16] - 2 * x[58], 0, 0, 0, 0, 2 * x[16] - 2 * x[4]],
        [0, 0, 0, 0, 2 * x[17] - 0.7, 0, 0, 0, 2 * x[17] - 2 * x[53], 0, 2 * x[17] - 2 * x[23], 0, 0, 0, 0, 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2 * x[18] - 1.0, 0, 0, 0, 2 * x[18] - 2 * x[54] - 2 * x[65], 0, 2 * x[18] - 2 * x[24], 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2 * x[19] - 1.5, 0, 0, 0, 2 * x[19] - 2 * x[55], 0, 2 * x[19] - 2 * x[25], 0, 0, 0, 0, 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[20] - 2 * x[62], -2 * x[14] + 2 * x[20], 0,
         2 * delta + 2 * x[20] - 2 * x[47], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[21] - 2 * x[63], -2 * x[15] + 2 * x[21], 0,
         2 * x[21] - 2 * x[48] - 2 * x[65], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[22] - 2 * x[64], -2 * x[16] + 2 * x[22], 0,
         2 * x[22] - 2 * x[49], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[17] + 2 * x[23], 2 * x[23] - 2 * x[8], 2 * x[23] - 2 * x[50], 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[18] + 2 * x[24], 2 * x[24] - 2 * x[9], 2 * x[24] - 2 * x[51], 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[19] + 2 * x[25], -2 * x[10] + 2 * x[25], 2 * x[25] - 2 * x[52], 0, 0, 0,
         0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2 * x[26] - 0.7, 0, 0, 0, 2 * x[26] - 2 * x[53], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 2 * x[26] - 2 * x[29], 0],
        [0, 0, 0, 2 * x[27] - 1.0, 0, 0, 0, 2 * x[27] - 2 * x[54], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 2 * x[27] - 2 * x[30], 0],
        [0, 0, 0, 2 * x[28] - 1.5, 0, 0, 0, 2 * x[28] - 2 * x[55], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 2 * x[28] - 2 * x[31] - 2 * x[66], 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[11] + 2 * x[29], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0,
         0, 0, 2 * x[29] - 2 * x[44], -2 * x[26] + 2 * x[29], 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[12] + 2 * x[30], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0,
         0, 0, 2 * x[30] - 2 * x[45], -2 * x[27] + 2 * x[30], 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[13] + 2 * x[31], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0,
         0, 0, 2 * x[31] - 2 * x[46], -2 * x[28] + 2 * x[31] + 2 * x[66], 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[32] - 2 * x[35],
         2 * delta + 2 * x[32] - 2 * x[47], 0, 0, 0, 0, 0, 2 * x[32] - 2 * x[38], 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[33] - 2 * x[36],
         2 * x[33] - 2 * x[48], 0, 0, 0, 0, 0, 2 * x[33] - 2 * x[39] + 2 * x[65], 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[34] - 2 * x[37],
         2 * x[34] - 2 * x[49], 0, 0, 0, 0, 0, 2 * x[34] - 2 * x[40], 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2 * x[35] - 2 * x[41], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[35] - 2 * x[50],
         -2 * x[32] + 2 * x[35], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2 * x[36] - 2 * x[42], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         2 * x[36] - 2 * x[51] + 2 * x[65], -2 * x[33] + 2 * x[36], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2 * x[37] - 2 * x[43], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[37] - 2 * x[52],
         -2 * x[34] + 2 * x[37], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[38] - 2 * x[41], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0,
         -2 * x[32] + 2 * x[38], 2 * x[38] - 2 * x[44], 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[39] - 2 * x[42] - 2 * x[65], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0,
         0, 0, 0, -2 * x[33] + 2 * x[39] - 2 * x[65], 2 * x[39] - 2 * x[45] - 2 * x[65], 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[40] - 2 * x[43] + 2 * x[66], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0,
         0, 0, 0, -2 * x[34] + 2 * x[40], 2 * x[40] - 2 * x[46], 0, 0, 0],
        [0, 0, 0, 0, 0, 2 * x[41] - 2 * x[53], -2 * x[35] + 2 * x[41], 0, 0, -2 * x[38] + 2 * x[41], 0, 0, 0, 0, 0, 0,
         0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2 * x[42] - 2 * x[54], -2 * x[36] + 2 * x[42], 0, 0, -2 * x[39] + 2 * x[42] + 2 * x[65], 0, 0,
         0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2 * x[43] - 2 * x[55], -2 * x[37] + 2 * x[43], 0, 0, -2 * x[40] + 2 * x[43] - 2 * x[66], 0, 0,
         0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[44] - 2 * x[59],
         0, 0,
         -2 * x[38] + 2 * x[44], -2 * x[29] + 2 * x[44], 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[45] - 2 * x[60],
         0, 0,
         -2 * x[39] + 2 * x[45] + 2 * x[65], -2 * x[30] + 2 * x[45], 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         2 * x[46] - 2 * x[61] + 2 * x[66], 0, 0, -2 * x[40] + 2 * x[46], -2 * x[31] + 2 * x[46], 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * delta - 2 * x[20] + 2 * x[47], 0, 0,
         -2 * delta - 2 * x[32] + 2 * x[47], -2 * delta + 2 * x[47] - 2 * x[56], 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[21] + 2 * x[48] + 2 * x[65], 0, 0,
         -2 * x[33] + 2 * x[48], 2 * x[48] - 2 * x[57], 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[22] + 2 * x[49], 0, 0,
         -2 * x[34] + 2 * x[49], 2 * x[49] - 2 * x[58], 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[23] + 2 * x[50], 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[50] - 2 * x[62], 0,
         -2 * x[35] + 2 * x[50], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[24] + 2 * x[51], 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[51] - 2 * x[63], 0,
         -2 * x[36] + 2 * x[51] - 2 * x[65], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[25] + 2 * x[52], 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[52] - 2 * x[64], 0,
         -2 * x[37] + 2 * x[52], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -2 * x[41] + 2 * x[53], 0, -2 * x[26] + 2 * x[53], -2 * x[17] + 2 * x[53], 0, 0, 0, 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -2 * x[42] + 2 * x[54], 0, -2 * x[27] + 2 * x[54], -2 * x[18] + 2 * x[54] + 2 * x[65], 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -2 * x[43] + 2 * x[55], 0, -2 * x[28] + 2 * x[55], -2 * x[19] + 2 * x[55], 0, 0, 0, 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         2 * delta - 2 * x[47] + 2 * x[56], 0,
         2 * x[56] - 2 * x[59], 0, -2 * x[14] + 2 * x[56], 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[48] + 2 * x[57], 0,
         2 * x[57] - 2 * x[60], 0, -2 * x[15] + 2 * x[57], 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[49] + 2 * x[58], 0,
         2 * x[58] - 2 * x[61] + 2 * x[66], 0, -2 * x[16] + 2 * x[58], 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         2 * delta - 2 * x[11] + 2 * x[59],
         -2 * x[56] + 2 * x[59], -2 * x[44] + 2 * x[59], 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[12] + 2 * x[60],
         -2 * x[57] + 2 * x[60], -2 * x[45] + 2 * x[60], 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         -2 * x[13] + 2 * x[61] - 2 * x[66], -2 * x[58] + 2 * x[61] - 2 * x[66], -2 * x[46] + 2 * x[61] - 2 * x[66], 0,
         0,
         0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * delta + 2 * x[62] - 2 * x[8], 0, 0, 0, 0, -2 * x[20] + 2 * x[62],
         0,
         -2 * x[50] + 2 * x[62], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * x[63] - 2 * x[9], 0, 0, 0, 0, -2 * x[21] + 2 * x[63], 0,
         -2 * x[51] + 2 * x[63], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * x[10] + 2 * x[64], 0, 0, 0, 0, -2 * x[22] + 2 * x[64], 0,
         -2 * x[52] + 2 * x[64], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -2 * x[18] + 2 * x[54] + 2 * x[65], -2 * x[39] + 2 * x[42] + 2 * x[65], 0, 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, -2 * x[21] + 2 * x[48] + 2 * x[65], 2 * x[36] - 2 * x[51] + 2 * x[65], 0, 0, 0, 0, 0, 0, 0,
         2 * x[33] - 2 * x[39] + 2 * x[65], -2 * x[39] + 2 * x[45] + 2 * x[65], 0, 0, 0],
        [0, 2 * x[2] + 2 * x[66] - 1.5, 0, 0, 0, 0, 0, 0, 0, 2 * x[40] - 2 * x[43] + 2 * x[66], 0, 0, 0, 0, 0, 0, 0, 0,
         0,
         0, 0, 0, 0, 0, 0, 0, 0, 2 * x[13] - 2 * x[61] + 2 * x[66], 2 * x[58] - 2 * x[61] + 2 * x[66],
         2 * x[46] - 2 * x[61] + 2 * x[66], 0, 0, 0, 0, -2 * x[28] + 2 * x[31] + 2 * x[66], 0]]
    return np.matrix(df).T



#Extended Ralphson-Newton method - perturbed starting point
# use extended Newton-Method to find zero's of polynomial system by randomly pertubating already known zero
def newton_pert(x0):
    x=x0
    i=0
    maxIt=1000
    
    
    eta = 10**(-5)                              # radius of pertubation
    bias = np.ones((len(x),1))                  # shift uniformly distribution to intervall -1 to 1
    
    pert1 =2*eta* (np.random.rand(len(x), 1)[0]-.5*bias[0])     
    x=x+pert1
    while(i<maxIt and np.linalg.norm(f(x))>precision):
        psi=df(x)
        inv=np.linalg.pinv(psi)                 # use Monroe-Pseudoinverse for Newton-Method to be applicalbe in an underdetermined system
        y=np.dot(inv,f(x))
        x=np.add(x,-y)
        i+=1
    return np.array(x)

# Extended Ralphson-Newton method - deterministic starting point
def newton(x0):
    i = 0
    x = x0

    maxIt = 10000

    while (i < maxIt and np.linalg.norm(f(x)) > precision):
        psi = df(x)
        inv = np.linalg.pinv(
            psi)  # use Monroe-Pseudoinverse for Newton-Method to be applicable in an underdetermined system
        y = np.array(np.dot(inv, f(x)))
        x = np.add(x, -y[0])
        i += 1

    return np.array(x)


#create a point cloud of samples of the algebraic variety locally
def sampling(number_of_samples,x):
    y = []
    for i in range(number_of_samples):
        y.append(list(newton(x)))
    return np.array(y)

#calculate mean value of all coordinate directions
def mean_vector(data):
    mean_vector = []
    for i in range(len(data[0])):
        mean_vector.append(np.mean(data[:,i]))
    return mean_vector

#calculate covariance matrix
def scatter_matrix(data):
    scatter  = np.zeros((len(data[0]),len(data[0])))
    mean = mean_vector(data)
    for i in range(len(data)):
        scatter += np.dot(data[i,:]-mean, (data[i,:]-mean).T)
    return 1(len(data)-1)*scatter

#calcualte Eigenvalue of covariance matrix
def eigenvalue(scat_matrix):
    return np.linalg.eig(scat_matrix)

# check if zero is isolated by surounding zero with starting points for Newton, such that every distinct new zero is closer to one of the starting points than the original zero
# cover a simplex plus origin with spheres of radius varying delta on centres of simplex plus origin with known zero centred at
# if there is any other zero in this simplex plus origin at least on vertex of the simplex would converge there under newton
def is_zero_dimensional(x0):
    m = len(x0)
    v = []
    v_centre = []
    eta = 10 ** (-8)  # radius of neighborhood

    # create centre point
    for i in range(m):
        v_centre.append(1 / (m + 1))

    # create points on vertices of tetrahedron
    for i in range(m + 1):
        # create coordinates of higherdim. tetrahedron
        coordinate = []
        for j in range(m):
            k = 0
            if (j + 1 == i): k = 1
            coordinate.append(k)
            # shift centre to x0
        # print(coordinate)
        coordinate = eta * np.array(coordinate) + np.array(x0) + eta * np.array(v_centre)  # translate tetrahedron to right place (ne
        v.append(list(coordinate))

    v.append((v_centre))

    # calculate closest zero to vertices of tetrahedron
    res = []

    for i in range(m + 1):
        res.append(newton((v[i])))

    var = 0
    for i in range(m + 1):
        var += np.linalg.norm(res[i] - x0)

    if (1 / (m + 1) * var < 10**(-15)):
        print(1 / (m + 1) * var)
        return True
    else:
        return False

#calculate dimension of algebraic variety by finding the nummber of non-zero Eigenvalues within some precision
def dimensionality_check(x0, sample_number):
    if(is_zero_dimensional(x0)):
        print('The system has dimension zero')
        return 0
    else:
        data = sampling(sample_number,x0)
        scat_matrix=scatter_matrix(data)
        eig_val=eigenvalue(scat_matrix)[0]
        print(eig_val)
        if all(eig < precision for eig in eig_val):
            print("The system is singular")
            return -1
            #Assumption: singularities always have tangent cone (translated affine cone (i.e. commplex case in Cox Little O'Shea))
            # and hence have 0 covariance eigenvalues

            #NEW! assumption might not be true, in the sense that there are singularieties with non-zero covariance.

            # if(is_Isolated(data,delta)): print('The system has dimension zero')
            # else: print('The system is singular')
        else:
            poss_dim =sum(eig_val>precision)
            print("the system has positive dimension smaller than "+str(poss_dim))
            return poss_dim
            #Assupmtion: Since all singularites are cone like all cases of positive covariance will be at a regular point

            # calculate_dimension(scat_matrix,poss_dim)

            #NEW! Slice system such that directions with zero variance can be checked on singularities

# # core of dimensionality_check(): Find true dimension by checking error of sample points to linear subspaces induced by PCA
# def calculate_dimension(data,poss_dim):
#     subspace=eigenvalue()
#     #check if a priori known solution is singularity
#     for n in range(poss_dim):
#         if (is_onSubspace(subspace)[0]):
#             return is_onSubspace[1]
#         else:
#             #subspace=
#             calculate_dimension(data,poss_dim)
#
# def is_onSubspace():
#


# intersect variety along eigenvectors of scatter matrix corresponding to vanishing eigenvalues
def is_Singular(data,delta):

    if(data ): return True
    else: return False

x= [
    0.75, 0.65, 0.0,
    0.6, 0.25,
    0.75, 0.35, 0.5,
    0.75, 0.9, 0.5,
    0.75, 0.1, 0.0,
    0.15, 0.5, 0.25,
    0.5, 0.75, 0.9,
    0.0, 0.75, 0.1,
    0.5, 0.75, 0.35,
    0.5, 0.25, 0.6,
    0.5, 0.25, 0.15,
    0.4, 0.0, 0.25,
    0.25, 0.15, 0.5,
    0.25, 0.85, 0.0,
    0.1, 0.0, 0.75,
    0.25, 0.4, 0.0,
    0.85, 0.0, 0.25,
    0.25, 0.6, 0.5,
    0.65, 0.0, 0.75,
    0.0, 0.25, 0.4,
    0.0, 0.25, 0.85,
    0.0, 0.75, 0.65,
    1, 1
]

dimensionality_check(x,100000)













