# 生成预测数据集结果
import pandas as pd
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.composition.alloy import WenAlloys
from matminer.featurizers.composition.element import ElementFraction
from matminer.featurizers.composition.composite import Meredig
import pandas as pd
# 创建 DataFrame
from strength_calculation import calculate_Strength
from pymatgen.core import Composition


def to_comp(x):
    try:
        return Composition.from_weight_dict(x.to_dict())
    except Exception as ex:
        print(x, ex)

def main():
    # 对析出相和惯习面进行编码
    df = pd.read_excel('materials_data_combined_500.xlsx')

    # print(df)
    elements = ['Al', 'Zn', 'Mn', 'Mg', 'Li', 'Sn', 'Ce', 'La', 'Ca', 'Si', 'Cu', 'Ni',
                'Fe', 'Bi', 'Ti', 'Sr', 'Sm', 'Sc', 'Gd', 'Y', 'Zr', 'Nd', 'Ag', 'Yb']
    weightPercentage = df[elements]

    compSeries = weightPercentage.apply(to_comp, axis = 1)
    print(compSeries)

    df_comp = pd.DataFrame({"composition": compSeries})
    # print('hello')
    # 初始化特征扩展器
    featurizer = ElementProperty.from_preset("magpie")
    # 应用特征扩展器
    df_comp = featurizer.featurize_dataframe(df_comp, "composition")
    df_comp = WenAlloys().featurize_dataframe(df_comp, "composition")
    df_comp = Meredig().featurize_dataframe(df_comp, "composition")
    for columnName, obj in df_comp.dtypes.items():
        if obj == "object":
            print(columnName)
            del df_comp[columnName]

    all = pd.concat([df_comp, df], axis = 1)
    all = calculate_Strength(all)
    all.to_excel('Target_data_500.xlsx')

if __name__ == '__main__':
    main()
