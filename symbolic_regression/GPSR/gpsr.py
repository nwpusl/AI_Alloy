import os
import pandas as pd
from pysr import PySRRegressor
from sympy import cos  # Import SymPy for custom operator definitions

def run_experiment(task_name, X, Y, data_file, target_name, base_directory="./"):
    # Ensure the base directory exists and navigate to it
    os.makedirs(base_directory, exist_ok=True)

    # Create a folder for the task within the base directory
    task_folder = os.path.join(base_directory, task_name)
    
    # Ensure that the task folder is created, without nesting
    os.makedirs(task_folder, exist_ok=True)
    
    # Change the working directory to the task folder
    os.chdir(task_folder)
    print(f"Working directory changed to: {os.getcwd()}")


    # Initialize the PySR model with customized settings
    model = PySRRegressor(
        maxsize=50,  # Maximum complexity
        niterations=100,  # Number of iterations
        binary_operators=["*", "+", "-", "/"],
        unary_operators=["square", "cube", "exp", "cos2(x)=cos(x)^2"],
        constraints={
            "/": (-1, 9),
            "square": 9,
            "cube": 9,
            "exp": 9,
        },
        nested_constraints={
            "square": {"square": 1, "cube": 1, "exp": 0},
            "cube": {"square": 1, "cube": 1, "exp": 0},
            "exp": {"square": 1, "cube": 1, "exp": 0},
        },
        extra_sympy_mappings={
        "inv": lambda x: 1 / x,  # Inverse operator
        "cos2": lambda x: cos(x)**2,  # Custom operator for `cos2`
    },       
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        early_stop_condition=(
            "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
        ),
        timeout_in_seconds=60 * 60 * 2,  # Timeout in seconds
    )

    # Fit the model and save the results in the task folder
    print(f"Starting experiment for task: {task_name}")
    model.fit(X, Y)
    print(f"Experiment {task_name} complete. Results saved to: {task_folder}")

def get_dataset_qf(data_file, task):
    Target = '屈服强度'
    if task == 'task1':
        names = ['Calculated Yield Strength', 'Grain Size', 'Calculated Grain Boundary', 'Interant electrons']
        Feature = names
        print('task:', task, Feature)
    if task == 'task2':
        names = ['Calculated Yield Strength', 'Grain Size', 'Calculated Grain Boundary', 'Interant electrons', 
                 'frac f valence electrons', 'MagpieData avg_dev MeltingT']
        Feature = names
        print('task:', task, Feature)
    if task == 'task3':
        names = ['Grain Size', 'Interant electrons', 'frac f valence electrons', 'MagpieData avg_dev MeltingT']
        Feature = names
        print('task:', task, Feature)
    if task == 'task4':
        names = ['Grain Size', 'Interant electrons', 'frac f valence electrons', 'MagpieData avg_dev MeltingT',
                 'mean AtomicRadius', 'Yang omega']
        Feature = names
        print('task:', task, Feature)
    X = pd.read_excel(data_file, header=0)[Feature]  # Ensure column headers are correctly read
    Y = pd.read_excel(data_file, header=0)[Target]
    return X, Y

def get_dataset_kl(data_file, task):
    Target = '抗拉强度 (UTS)'
    if task == 'task1':
        names = ['Grain Size', 'Calculated Yield Strength', 'MagpieData mean GSvolume_pa', 'Interant electrons']
        Feature = names
        print('task:', task, Feature)
    if task == 'task2':
        names = ['Grain Size', 'Calculated Yield Strength', 'MagpieData mean GSvolume_pa', 'Interant electrons', 
                 'Interant d electrons', 'Calculated Grain Boundary']
        Feature = names
        print('task:', task, Feature)
    if task == 'task3':
        names = ['Grain Size', 'MagpieData mean GSvolume_pa', 'Interant electrons', 'Interant d electrons']
        Feature = names
        print('task:', task, Feature)
    if task == 'task4':
        names = ['Grain Size', 'MagpieData mean GSvolume_pa', 'Interant electrons', 'Interant d electrons', 
                 'MagpieData range GSvolume_pa', 'Mixing enthalpy']
        Feature = names
        print('task:', task, Feature)
    X = pd.read_excel(data_file, header=0)[Feature]  # Ensure column headers are correctly read
    Y = pd.read_excel(data_file, header=0)[Target]
    return X, Y

# Task and dataset configuration
train_data_file_qf = r'G:\AIMS_Lab_final\my_project\AAA-MgAlloy\AI_Alloy\Yield_Strength\train_set_new.xlsx'
train_data_file_kl = r'G:\AIMS_Lab_final\my_project\AAA-MgAlloy\AI_Alloy\Tensile_Strength\train_set_new.xlsx'

task_list = [f'task{i}' for i in range(1, 5)]  # Tasks: task1, task2, task3, task4

# Run experiments for QF dataset
for task in task_list:
    X_qf, Y_qf = get_dataset_qf(train_data_file_qf, task)
    run_experiment(f"qf_{task}", X_qf, Y_qf, train_data_file_qf, '屈服强度')

# Run experiments for KL dataset
for task in task_list:
    X_kl, Y_kl = get_dataset_kl(train_data_file_kl, task)
    run_experiment(f"kl_{task}", X_kl, Y_kl, train_data_file_kl, '抗拉强度 (UTS)')
