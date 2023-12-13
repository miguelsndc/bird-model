from dataclasses import dataclass 
import os

@dataclass
class Target:
    class_name: str
    paths: list[str]


def build_report_target(path: str): #constroi um array de objetos contendo os caminhos de cada imagem e sua respectiva classe
    testing_data: list[Target] = []
    for root, dirs, files in os.walk(path):
        if len(dirs) > 0:
            continue

        current_class = root.split("\\")[-1]
        files = [os.path.join(root, file) for file in files]
        testing_data.append(Target(current_class, files))
    return testing_data

if __name__ == "__main__":
    data = build_report_target(os.path.join(os.getcwd(), "images"))
