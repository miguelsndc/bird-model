from mlforkids import MLforKidsImageProject
import os
import json
import statistics
from report import build_report_target

key = "e37a4fa0-9919-11ee-86a2-1d9fdd4d6d17571cd6b7-3822-4ec0-96bc-e9d688796e38"
path = os.path.join(os.getcwd(), "images")

class JsonStorage:
    def __init__(self, path) -> None:
        self.db_path = path

    def load(self):
        with open(self.db_path, "r") as file:
            source = json.loads(file.read())
            return source

    def save(self, object):
        with open(self.db_path, "w") as file:
            json.dump(object, file, indent=4)

    def clear(self):
        with open(self.db_path, 'w') as file:
            file.write("{'runs': []}")

storage = JsonStorage(os.path.join(os.getcwd(), "results.json"))

def create_model(key: str):
    model = MLforKidsImageProject(key)
    model.train_model()
    return model

def generate_accuracy_report(target_path: str):
    model = create_model(key)
    targets = build_report_target(target_path)
    results = []
    for target in targets:
        accuracies = []
        for path in target.paths:
            prediction = model.prediction(path)
            accuracies.append(prediction["confidence"])
        result = {
            "class_name": target.class_name,
            "accuracies": accuracies.copy(),
            "accuracy_average": statistics.mean(accuracies),
            "accuracy_variance": statistics.variance(accuracies),
            "accuracy_deviation": statistics.stdev(accuracies)
        }
        results.append(result)

    return results

results = generate_accuracy_report(path)
metadata = {
    "average_accuracy": statistics.mean([r["accuracy_average"] for r in results]),
    "average_variance": statistics.mean([r["accuracy_variance"] for r in results]),
    "average_deviation": statistics.mean([r["accuracy_deviation"] for r in results])
}
storage.save({"metadata": metadata, "results": results})
