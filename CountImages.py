import os
import matplotlib.pyplot as plt
import pandas

image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

folders = {}

def count(directory_path):
    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)

        count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(image_extensions):
                count += 1

        folders[folder_name] = count
    
    for folder_name, image_count in folders.items():
        print(f"Folder: {folder_name}, Image Count: {image_count}")
    

    emotions = list(folders.keys())
    counts = list(folders.values())
    plt.bar(emotions, counts, color = 'blue', width = 0.7)
    plt.xlabel("Emotions")
    plt.ylabel("Counts")
    plt.show()
