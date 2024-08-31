import os.path
import os
from pymatgen.core import Composition
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.composition.alloy import WenAlloys
from matminer.featurizers.composition.element import ElementFraction
from matminer.featurizers.composition.composite import Meredig
import time
import math
import numpy as np
import itertools
import pandas as pd
file_count = 5
# 选取24种元素的不同分布构建质量比数据，最终生成44w条数据
elements_majority = {
    'Al': np.concatenate((np.arange(0, 10, 1), np.arange(10, 20, 3)), axis=0),
    'Zn': np.concatenate((np.arange(0, 2, 1), np.arange(2, 20, 5)), axis=0),
    'Gd': np.concatenate((np.arange(0, 5, 1), np.arange(5, 20, 5)), axis=0),
    'Y': np.concatenate((np.arange(0, 5, 1), np.arange(5, 20, 5)), axis=0),
    'Sm': np.concatenate((np.arange(0, 1, 0.3), np.arange(1, 2, 0.5), np.arange(2, 15, 5)), axis=0),
    'Mn': np.concatenate((np.arange(0, 1, 0.3), np.arange(1, 2, 0.5), np.arange(2, 5, 1.5)), axis=0),
    'Zr': np.concatenate((np.arange(0, 1, 0.3), np.arange(1, 5, 1)), axis=0)
}
elements_minority={
    'Sc': np.concatenate((np.arange(0,4,1), np.arange(4,15,5)), axis=0),
    'Li': np.concatenate((np.arange(0,4,1), np.arange(4,10,5)), axis=0),
    'Sn': np.concatenate((np.arange(0,4,1), np.arange(4,15,4)), axis=0),
    'Ce': np.concatenate((np.arange(0,1,0.3),  np.arange(1,5,1.5)), axis=0),
    'Sr': np.concatenate((np.arange(0,1,0.3), np.arange(1,5,1.5)), axis=0),
    'Nd': np.concatenate((np.arange(0,1,0.3), np.arange(1,5,1.5)), axis=0),
    'Yb': np.concatenate((np.arange(0,1,0.3), np.arange(1,5,1.5)), axis=0),
    'Bi': np.concatenate((np.arange(0,1,0.3), np.arange(1,5,1.5)), axis=0),
    'Ca': np.concatenate((np.arange(0,1,0.3), np.arange(1,5,1.5)), axis=0),
    'La': np.concatenate((np.arange(0,0.3,0.1), np.arange(0.3,1.5,0.5)), axis=0),
    'Ti': np.concatenate((np.arange(0,0.1,0.03), np.arange(0.1,0.5,0.2)), axis=0),
    'Fe': np.concatenate((np.arange(0,0.1,0.03), np.arange(0.1,0.2,0.06)), axis=0),
    'Cu': np.concatenate((np.arange(0,0.005,0.002), np.arange(0.005,0.012,0.003)), axis=0),
    'Si': np.concatenate((np.arange(0,1,0.3), np.arange(1,5,2)), axis=0),
    'Ag': np.arange(0,2,0.4),
    'Ni': np.concatenate((np.arange(0,0.005,0.002), np.arange(0.005,0.012,0.004)), axis=0)
}
element_order = ['Al', 'Zn', 'Mn', 'Mg', 'Li', 'Sn', 'Ce', 'La', 'Ca', 'Si', 'Cu', 'Ni', 'Fe', 'Bi', 'Ti', 'Sr', 'Sm', 'Sc', 'Gd', 'Y', 'Zr', 'Nd', 'Ag', 'Yb']
def get_combinations(elements_majority, elements_minority):
    cnt = 1
    for r in range(1, 5):  # 选择从1到4个元素
        # 当仅选择1个或0个majority元素时，考虑minority元素的选择
        if r <= 1:
            for major_subset in itertools.combinations(elements_majority, r):
                for minor_r in range(3):  # 选择从0到2个minority元素
                    for minor_subset in itertools.combinations(elements_minority, minor_r):
                        # 合并majority和minority元素
                        subset = major_subset + minor_subset
                        lists = [elements_majority.get(element) if element in elements_majority else elements_minority.get(element) for element in subset]
                        for product_combination in itertools.product(*lists):
                            if sum(product_combination) <= 20:
                                cnt += 1
                                if cnt % 10000 == 0:
                                    print('产生第', cnt / 10000, 'w条数据')
                                combination = {element: value for element, value in zip(subset, product_combination)}
                                combination['Mg'] = 100 - sum(product_combination)
                                for element in element_order:
                                    if element not in combination:
                                        combination[element] = 0
                                yield {element: combination[element] for element in element_order}
        # 当选择2到4个majority元素时，保持原有处理方式不变
        else:
            for subset in itertools.combinations(elements_majority, r):
                lists = [elements_majority.get(element) for element in subset]
                for product_combination in itertools.product(*lists):
                    if sum(product_combination) <= 20:
                        cnt += 1
                        if cnt % 10000 == 0:
                            print('产生第', cnt / 10000, 'w条数据')
                        combination = {element: value for element, value in zip(subset, product_combination)}
                        combination['Mg'] = 100 - sum(product_combination)
                        for element in element_order:
                            if element not in combination:
                                combination[element] = 0
                        yield {element: combination[element] for element in element_order}
# 示例使用修改后的函数
combinations_list = list(get_combinations(elements_majority, elements_minority))
df_combinations = pd.DataFrame(combinations_list)
# 检查生成的条数，决定是否需要分批
print(f"总共生成了{len(df_combinations)}条组合。")
batch_size = 100000  # 根据你的需要可能需要调整
num_batches = len(combinations_list) // batch_size + 1
for i in range(num_batches):
    print(f'输出前{i+1}0w条数据的结果')
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, len(combinations_list))
    df_batch = df_combinations.iloc[start_index:end_index]
    filename = f"combinations0318_{i+1}.xlsx"
    df_batch.to_excel(filename, index=False)
    print(f"数据已保存到 {filename}")
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
        strength += ((Bi_Table[element] ** 1.5) * row.get(f"{element} fraction", 0))
    return strength
def calculate_jingjie(row):
    if row["晶粒尺寸"] == 0:
        return 0
    strength = 210
    for element in elements:
        strength += (qh_Table[element] * row.get(f"{element} fraction", 0))
    return strength / math.sqrt(row["晶粒尺寸"])
Grain_Size={'零':0,'小尺寸':2,'中位数':10,'大尺寸':42}
Habit_Panel={'基面':0,'基面+杂质':1,'基面+杂质+柱面':2,"基面+柱面":3,"无":4,"杂质":5,"杂质+柱面":6,"柱面":7}
Precipitation_phase_distribution={'无':0,'晶内':1,'晶界':2,'晶界+晶内':3,'晶粒':4}
for excel_index in range(1, 6):
    tic = time.time()
    print(f'开始处理第{excel_index}个表格的数据')
    # 读取原始数据
    df = pd.read_excel(f'original_mass_results/combinations0318_{excel_index}.xlsx')
    # 复制并分配晶粒尺寸
    replicated_data = []
    for index, row in df.iterrows():
        for grain_size_key, grain_size_value in Grain_Size.items():
            row_copy = row.copy()
            row_copy['晶粒尺寸'] = grain_size_value  # 分配不同的晶粒尺寸
            replicated_data.append(row_copy)
    df_replicated = pd.DataFrame(replicated_data)
    # 分配随机特征 - 惯习面和析出相分布
    df_replicated['惯习面'] = np.random.choice(list(Habit_Panel.values()), len(df_replicated))
    df_replicated['析出相分布'] = np.random.choice(list(Precipitation_phase_distribution.values()), len(df_replicated))
    df1=df_replicated[['Al', 'Zn', 'Mn', 'Mg', 'Li', 'Sn', 'Ce', 'La', 'Ca', 'Si', 'Cu', 'Ni','Fe', 'Bi', 'Ti', 'Sr', 'Sm', 'Sc', 'Gd', 'Y', 'Zr', 'Nd', 'Ag', 'Yb']]
    # 将复制的结果/不含晶粒尺寸的复制进去
    if not os.path.exists('original_mass_results_expanded'):
        os.mkdir('original_mass_results_expanded')
    df1.to_excel(f'original_mass_results_expanded/origin_Alloy_dataset{excel_index}.xlsx')
    # 开始生成新的表格
    weightPercentage = df_replicated[['Al', 'Zn', 'Mn', 'Mg', 'Li', 'Sn', 'Ce', 'La', 'Ca', 'Si', 'Cu', 'Ni','Fe', 'Bi', 'Ti', 'Sr', 'Sm', 'Sc', 'Gd', 'Y', 'Zr', 'Nd', 'Ag', 'Yb']]
    compSeries = weightPercentage.apply(lambda x: Composition.from_weight_dict(x.to_dict()), axis=1)
    # print(weightPercentage.iloc[0])
    # print(compSeries[0])
    # 计算matminer特征
    df = pd.DataFrame({"composition": compSeries})
    print('开始生成pymatgent特征')
    featurizer = ElementProperty.from_preset("magpie")
    df = featurizer.featurize_dataframe(df, "composition")
    df = WenAlloys().featurize_dataframe(df, "composition")
    df = ElementFraction().featurize_dataframe(df, "composition")
    df = Meredig().featurize_dataframe(df, "composition")
    df["晶粒尺寸"]=df_replicated["晶粒尺寸"]
    df["惯习面"]=df_replicated['惯习面']
    df['析出相分布']=df_replicated['析出相分布']
    df["计算固溶"] =df[[f"{i} fraction" for i in elements]].apply(calculate_solubility_strength, axis=1)
    df["计算晶界"] = df[[f"{i} fraction" for i in elements] + ["晶粒尺寸"]].apply(calculate_jingjie, axis=1)
    df["计算总强度"] = df["计算固溶"] + df["计算晶界"]
    df=df[['MagpieData minimum Number', 'MagpieData maximum Number', 'MagpieData range Number', 'MagpieData mean Number', 'MagpieData avg_dev Number', 'MagpieData mode Number', 'MagpieData minimum MendeleevNumber', 'MagpieData maximum MendeleevNumber', 'MagpieData range MendeleevNumber', 'MagpieData mean MendeleevNumber', 'MagpieData avg_dev MendeleevNumber', 'MagpieData mode MendeleevNumber', 'MagpieData minimum AtomicWeight', 'MagpieData maximum AtomicWeight', 'MagpieData range AtomicWeight', 'MagpieData mean AtomicWeight', 'MagpieData avg_dev AtomicWeight', 'MagpieData mode AtomicWeight', 'MagpieData minimum MeltingT', 'MagpieData maximum MeltingT', 'MagpieData range MeltingT', 'MagpieData mean MeltingT', 'MagpieData avg_dev MeltingT', 'MagpieData mode MeltingT', 'MagpieData minimum Column', 'MagpieData maximum Column', 'MagpieData range Column', 'MagpieData mean Column', 'MagpieData avg_dev Column', 'MagpieData mode Column', 'MagpieData minimum Row', 'MagpieData maximum Row', 'MagpieData range Row', 'MagpieData mean Row', 'MagpieData avg_dev Row', 'MagpieData mode Row', 'MagpieData minimum CovalentRadius', 'MagpieData maximum CovalentRadius', 'MagpieData range CovalentRadius', 'MagpieData mean CovalentRadius', 'MagpieData avg_dev CovalentRadius', 'MagpieData mode CovalentRadius', 'MagpieData minimum Electronegativity', 'MagpieData maximum Electronegativity', 'MagpieData range Electronegativity', 'MagpieData mean Electronegativity', 'MagpieData avg_dev Electronegativity', 'MagpieData mode Electronegativity', 'MagpieData minimum NsValence', 'MagpieData maximum NsValence', 'MagpieData range NsValence', 'MagpieData mean NsValence', 'MagpieData avg_dev NsValence', 'MagpieData mode NsValence', 'MagpieData minimum NpValence', 'MagpieData maximum NpValence', 'MagpieData range NpValence', 'MagpieData mean NpValence', 'MagpieData avg_dev NpValence', 'MagpieData mode NpValence', 'MagpieData minimum NdValence', 'MagpieData maximum NdValence', 'MagpieData range NdValence', 'MagpieData mean NdValence', 'MagpieData avg_dev NdValence', 'MagpieData mode NdValence', 'MagpieData minimum NfValence', 'MagpieData maximum NfValence', 'MagpieData range NfValence', 'MagpieData mean NfValence', 'MagpieData avg_dev NfValence', 'MagpieData mode NfValence', 'MagpieData minimum NValence', 'MagpieData maximum NValence', 'MagpieData range NValence', 'MagpieData mean NValence', 'MagpieData avg_dev NValence', 'MagpieData mode NValence', 'MagpieData minimum NsUnfilled', 'MagpieData maximum NsUnfilled', 'MagpieData range NsUnfilled', 'MagpieData mean NsUnfilled', 'MagpieData avg_dev NsUnfilled', 'MagpieData mode NsUnfilled', 'MagpieData minimum NpUnfilled', 'MagpieData maximum NpUnfilled', 'MagpieData range NpUnfilled', 'MagpieData mean NpUnfilled', 'MagpieData avg_dev NpUnfilled', 'MagpieData mode NpUnfilled', 'MagpieData minimum NdUnfilled', 'MagpieData maximum NdUnfilled', 'MagpieData range NdUnfilled', 'MagpieData mean NdUnfilled', 'MagpieData avg_dev NdUnfilled', 'MagpieData mode NdUnfilled', 'MagpieData minimum NfUnfilled', 'MagpieData maximum NfUnfilled', 'MagpieData range NfUnfilled', 'MagpieData mean NfUnfilled', 'MagpieData avg_dev NfUnfilled', 'MagpieData mode NfUnfilled', 'MagpieData minimum NUnfilled', 'MagpieData maximum NUnfilled', 'MagpieData range NUnfilled', 'MagpieData mean NUnfilled', 'MagpieData avg_dev NUnfilled', 'MagpieData mode NUnfilled', 'MagpieData minimum GSvolume_pa', 'MagpieData maximum GSvolume_pa', 'MagpieData range GSvolume_pa', 'MagpieData mean GSvolume_pa', 'MagpieData avg_dev GSvolume_pa', 'MagpieData mode GSvolume_pa', 'MagpieData minimum GSbandgap', 'MagpieData maximum GSbandgap', 'MagpieData range GSbandgap', 'MagpieData mean GSbandgap', 'MagpieData avg_dev GSbandgap', 'MagpieData mode GSbandgap', 'MagpieData minimum GSmagmom', 'MagpieData maximum GSmagmom', 'MagpieData range GSmagmom', 'MagpieData mean GSmagmom', 'MagpieData avg_dev GSmagmom', 'MagpieData mode GSmagmom', 'MagpieData minimum SpaceGroupNumber', 'MagpieData maximum SpaceGroupNumber', 'MagpieData range SpaceGroupNumber', 'MagpieData mean SpaceGroupNumber', 'MagpieData avg_dev SpaceGroupNumber', 'MagpieData mode SpaceGroupNumber', 'Yang delta', 'Yang omega', 'APE mean', 'Radii local mismatch', 'Radii gamma', 'Configuration entropy', 'Atomic weight mean', 'Total weight', 'Lambda entropy', 'Electronegativity delta', 'Electronegativity local mismatch', 'VEC mean', 'Mixing enthalpy', 'Mean cohesive energy', 'Interant electrons', 'Interant s electrons', 'Interant p electrons', 'Interant d electrons', 'Interant f electrons', 'Shear modulus mean', 'Shear modulus delta', 'Shear modulus local mismatch', 'Shear modulus strength model', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'H fraction', 'He fraction', 'Li fraction', 'Be fraction', 'B fraction', 'C fraction', 'N fraction', 'O fraction', 'F fraction', 'Ne fraction', 'Na fraction', 'Mg fraction', 'Al fraction', 'Si fraction', 'P fraction', 'S fraction', 'Cl fraction', 'Ar fraction', 'K fraction', 'Ca fraction', 'Sc fraction', 'Ti fraction', 'V fraction', 'Cr fraction', 'Mn fraction', 'Fe fraction', 'Co fraction', 'Ni fraction', 'Cu fraction', 'Zn fraction', 'Ga fraction', 'Ge fraction', 'As fraction', 'Se fraction', 'Br fraction', 'Kr fraction', 'Rb fraction', 'Sr fraction', 'Y fraction', 'Zr fraction', 'Nb fraction', 'Mo fraction', 'Tc fraction', 'Ru fraction', 'Rh fraction', 'Pd fraction', 'Ag fraction', 'Cd fraction', 'In fraction', 'Sn fraction', 'Sb fraction', 'Te fraction', 'I fraction', 'Xe fraction', 'Cs fraction', 'Ba fraction', 'La fraction', 'Ce fraction', 'Pr fraction', 'Nd fraction', 'Pm fraction', 'Sm fraction', 'Eu fraction', 'Gd fraction', 'Tb fraction', 'Dy fraction', 'Ho fraction', 'Er fraction', 'Tm fraction', 'Yb fraction', 'Lu fraction', 'Hf fraction', 'Ta fraction', 'W fraction', 'Re fraction', 'Os fraction', 'Ir fraction', 'Pt fraction', 'Au fraction', 'Hg fraction', 'Tl fraction', 'Pb fraction', 'Bi fraction', 'Po fraction', 'At fraction', 'Rn fraction', 'Fr fraction', 'Ra fraction', 'Ac fraction', 'Th fraction', 'Pa fraction', 'U fraction', 'Np fraction', 'Pu fraction', 'Am fraction', 'Cm fraction', 'Bk fraction', 'Cf fraction', 'Es fraction', 'Fm fraction', 'Md fraction', 'No fraction', 'Lr fraction', 'mean AtomicWeight', 'mean Column', 'mean Row', 'range Number', 'mean Number', 'range AtomicRadius', 'mean AtomicRadius', 'range Electronegativity', 'mean Electronegativity', 'avg s valence electrons', 'avg p valence electrons', 'avg d valence electrons', 'avg f valence electrons', 'frac s valence electrons', 'frac p valence electrons', 'frac d valence electrons', 'frac f valence electrons', '晶粒尺寸', '惯习面', '析出相分布', '计算固溶', '计算晶界', '计算总强度']]
    #获得'晶粒尺寸', '惯习面', '析出相分布'
    if not os.path.exists('alloy_dataset_results_expanded'):
        os.mkdir('alloy_dataset_results_expanded')
    df.to_excel(f'alloy_dataset_results_expanded/Alloy_dataset{excel_index}.xlsx')
    print(f'第{excel_index}个数据集内容生成结束')
    # print(df.shape)
    # print(df)
    toc=time.time()
    # print(df[:100,:])
    print(f'生成pymatgen特征总共花费{toc-tic}s')