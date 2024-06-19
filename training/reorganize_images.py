import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Diretórios de origem e destino
source_dir = 'data/food-101/images'
target_dir = 'data'

train_dir = os.path.join(target_dir, 'train')
test_dir = os.path.join(target_dir, 'test')

# Criar diretórios de treinamento e teste
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Lista de classes
classes = sorted(os.listdir(source_dir))

# Função para mover arquivos
def move_files(file_list, source_dir, target_dir, class_name):
    target_class_dir = os.path.join(target_dir, class_name)
    os.makedirs(target_class_dir, exist_ok=True)
    for file in file_list:
        shutil.copy(os.path.join(source_dir, class_name, file), target_class_dir)

# Iterar sobre cada classe
for class_name in classes:
    class_dir = os.path.join(source_dir, class_name)
    if os.path.isdir(class_dir):
        files = os.listdir(class_dir)
        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
        
        move_files(train_files, source_dir, train_dir, class_name)
        move_files(test_files, source_dir, test_dir, class_name)

print("Dados organizados com sucesso.")
