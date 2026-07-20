"""Interactive visualization script for relational vs coordinate triplets."""

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Visualize relational vs coordinate triplets from annotation CSV.")
    parser.add_argument("csv_path", type=str, help="Path to the annotation.csv file.")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: file not found at {csv_path}")
        return

    dataset_dir = csv_path.parent

    triplets = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            triplets.append(row)

    if not triplets:
        print("Error: No samples found in the annotation CSV.")
        return

    print(f"Loaded {len(triplets)} triplets from {csv_path}.")
    print("Instructions:")
    print(" - Press RIGHT arrow or SPACE to go to the next triplet.")
    print(" - Press LEFT arrow to go to the previous triplet.")
    print(" - Press 'q' to quit.")

    current_idx = [0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    def draw_triplet(idx):
        row = triplets[idx]
        sample_id = row.get("SampleID", str(idx))
        root_orientation = row.get("RootOrientation", "N/A")
        
        conditions = [
            ("Basis", row["BasisPath"]),
            ("Coordinate Change", row["CoordinateChangePath"]),
            ("Relation Change", row["RelationChangePath"])
        ]
        
        for ax, (title, relative_path) in zip(axes, conditions):
            ax.clear()
            img_path = dataset_dir / relative_path
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f"Image not found:\n{relative_path}", 
                        ha='center', va='center', color='red', fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis("off")
            
        fig.suptitle(f"Triplet {idx + 1}/{len(triplets)} (Sample ID: {sample_id}, Root: {root_orientation})", 
                     fontsize=14, fontweight='bold')
        plt.draw()

    def on_key(event):
        if event.key in ('right', ' ', 'enter'):
            current_idx[0] = (current_idx[0] + 1) % len(triplets)
            draw_triplet(current_idx[0])
        elif event.key in ('left', 'backspace'):
            current_idx[0] = (current_idx[0] - 1) % len(triplets)
            draw_triplet(current_idx[0])
        elif event.key == 'q':
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw_triplet(0)
    plt.show()


if __name__ == "__main__":
    main()
