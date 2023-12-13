from mlforkids import MLforKidsImageProject
import os
import json

key = "e37a4fa0-9919-11ee-86a2-1d9fdd4d6d17571cd6b7-3822-4ec0-96bc-e9d688796e38"

def load(path: str):
    with open(path, "r") as file:
        source = json.loads(file.read())
        return source

def save(db, path: str):
    with open(path, "w") as file:
        json.dump(db, file, indent=4)


db_path = os.path.join(os.getcwd(), "predictions.json")
source = load(db_path)

def sync():
    save(source, db_path)

def create_model(key: str):
    model = MLforKidsImageProject(key)
    model.train_model()
    return model

def get_image_paths():
    images_dir = os.path.join(os.getcwd(), "images\\")
    files = [{"path": os.path.join(images_dir, path), "target": path} for path in os.listdir(images_dir)]
    return files

def clear(path: str):
    with open(path, 'w') as file:
        file.write("{'runs': []}")

def write_predictions(model: MLforKidsImageProject):
    files = get_image_paths()
    for image in files:
        demo = model.prediction(image['path'])
        run = {
            "target": image['target'],
            "class_name": demo["class_name"],
            "confidence": demo['confidence']
        }
        source['runs'].append(run)
    sync()

def display():
    for i, run in enumerate(source['runs']):
        print(f"RUN No {i + 1}")
        print(f"target: {run['target']}")
        print(f"class: {run['class_name']}")
        print(f"confidence: {run['confidence']:.2f}%")
        print('-' * 40)

model = create_model(key)

while True:
    print("-" * 40)
    print("press (d) to display prediction data")
    print("press (t) to make new predictions in images/ folder")
    print("press (c) to clear predictions and start all over")
    print("press (q) to quit")
    print(">>>>", end=" ")
    action = input("").strip().lower()
    print("")

    predictions_file = os.path.join(os.getcwd(), 'predictions.txt')

    match action:
        case "d":
            display()
        case "t":
            write_predictions(model)
        case "c":
            clear(db_path)
        case "q":
            break
        case _:
            continue






