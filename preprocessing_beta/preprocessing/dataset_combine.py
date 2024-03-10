import os
import csv
import random
import shutil


def read_csv(filenames):
    real = []
    fake = []
    for filename in filenames:
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[1] == "REAL":
                    real.append(row)
                if row[1] == "FAKE":
                    fake.append(row)
             
    return real,fake

def combine_folders(folder1, folder2, target_folder):
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # Iterate through files in folder1
    for file in files1:
        file_path = os.path.join(folder1, file)
        shutil.copy(file_path, target_folder)

    # Iterate through files in folder2
    for file in files2:
        file_path = os.path.join(folder2, file)
        shutil.copy(file_path, target_folder)

# File and Folder Paths
num_folders = 2
folder1 = ['/home/jenish/dfproj/filtered_dataset_combined_main/dataset/videos_112','/home/jenish/dfproj/filtered_dataset_combined_main/dataset/videos_224']
folder2 = ['/home/jenish/dfproj/preprocessing_beta/data_root/dset0/videos_112','/home/jenish/dfproj/preprocessing_beta/data_root/dset0/videos_224']
metadata_files = ['/home/jenish/dfproj/filtered_dataset_combined_main/dataset/metadata_filtered.csv', '/home/jenish/dfproj/preprocessing_beta/data_root/dset0/finalmetadata.csv']
target_folders = ['/home/jenish/dfproj/main/videos_112','/home/jenish/dfproj/main/videos_224']
target_metadata = '/home/jenish/dfproj/main'

for i in range(num_folders):
    combine_folders(folder1[i], folder2[i], target_folders[i])
   
data = read_csv(metadata_files)
aggregated_list = data[0] + data[1]
random.shuffle(aggregated_list)
print(len(aggregated_list))

with open(f'{target_metadata}/metadata_filtered.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for elem in aggregated_list:
        writer.writerow(elem)