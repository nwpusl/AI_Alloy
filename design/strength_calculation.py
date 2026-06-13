import math
elements = ['Al', 'Zn', 'Mn', 'Mg', 'Li', 'Sn', 'Ce', 'La', 'Ca', 'Si', 'Cu', 'Ni',
'Fe', 'Bi', 'Ti', 'Sr', 'Sm', 'Sc', 'Gd', 'Y', 'Zr', 'Nd', 'Ag', 'Yb',]
Bi_Table = {
    'Ag': 92.2,
    'Al': 120,
    'Bi': 100,  # 估计
    'Ca': 402.56,
    'Ce': 2077,
    'Cu': 100,  # 估计
    'Fe': 429.72,
    'Gd': 900,
    'La': 1698,
    'Li': 159.1,
    'Mg': 0,  # 规定为0
    'Mn': 158,
    'Nd': 813,
    'Ni': 524.93,
    'Sc': 477,
    'Si': 408.94,
    'Sm': 480.94,
    'Sn': 280,
    'Sr': 670.19,
    'Ti': 100,  # 没有记录！ V错误
    'Y': 800,
    'Yb': 432.87,
    'Zn': 40,
    'Zr': 273
}
qh_Table = {
    'Ag': 1000,  # 估计
    'Al': 2000,
    'Bi': 1000,  # 估计
    'Ca': 1000,  # 估计
    'Ce': 50827,
    'Cu': 1000,  # 估计
    'Fe': 1000,  # 估计
    'Gd': 5000,
    'La': 3322,
    'Li': 1000,  # 估计
    'Mg': 0,  # 应该是210，但是这里不算
    'Mn': 2000,
    'Nd': 20390,
    'Ni': 1000,  # 估计
    'Sc': 1570,
    'Si': 18000,
    'Sm': 1000,  # 估计
    'Sn': 3000,
    'Sr': 1000,  # 估计
    'Ti': 1000,  # 没有记录！ V错误
    'Y': 1000,
    'Yb': 1000,  # 估计
    'Zn': 18000,
    'Zr': 2038
}
# from pprint import pprint
# pprint(dict(zip(elements, Bi_Table)))


def calculate_solubility_strength(row):
    strength = 0
    for element in elements:
        # 对每个元素应用公式：Bi系数**1.5 * 比例
        strength += ((Bi_Table[element] ** 1.5) * row.get(f"{element}", 0))
    return strength**(2/3)
    # return strength

def calculate_jingjie(row):
    # 通过数据处理不再有晶粒尺寸为0的数据了
    if row["Grain Size"] == 0:
        return 0
    strength = 210
    for element in elements:
        strength += (qh_Table[element] * row.get(f"{element}", 0))
    return strength / math.sqrt(row["Grain Size"])



def calculate_Strength(FULL):
    """
    输入excel数据文件
    :return: 计算固溶，计算晶界，计算总强度
    """
    FULL["Calculated Solid Solution"] = FULL[[f"{i}" for i in elements]].apply(calculate_solubility_strength, axis=1)

    FULL["Calculated Grain Boundary"] = FULL[[f"{i}" for i in elements] + ["Grain Size"]].apply(calculate_jingjie, axis=1)

    FULL["Calculated Yield Strength"] = FULL["Calculated Solid Solution"] + FULL["Calculated Grain Boundary"]
    FULL["Calculated Yield Strength"] = FULL["Calculated Yield Strength"] + 15

    return FULL


