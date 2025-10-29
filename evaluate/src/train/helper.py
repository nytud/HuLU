import json
import os


def write_submission(task, predictions_data, output_dir) -> None:
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"{task}_predicted_labels.json"
    predictions_path = os.path.join(output_dir, base_filename)

    file_index = 1
    while os.path.exists(predictions_path):
        predictions_path = os.path.join(
            output_dir, f"{task}_predicted_labels_{file_index}.json"
        )
        file_index += 1

    with open(predictions_path, "w", encoding="utf8") as f:
        json.dump(predictions_data, f, indent=4)

    print(f"\n######### Saved predicted labels to {predictions_path} #########\n")
