import csv
import random
import os
import glob

def read_csv(filename):
    real = []
    fake = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[1] == "REAL":
                real.append(row)
            if row[1] == "FAKE":
                fake.append(row)
             
    return real,fake

def fake_list_filter(real, fake):
    filtered_fake_list = []
    real_copy = [x[0] for x in real]
    for item in fake:
        if f'{item[2]}.mp4' in real_copy:
            filtered_fake_list.append(item)
            real_copy.remove(f'{item[2]}.mp4')
    return filtered_fake_list





# File Paths
filename = '/home/jenish/dfproj/preprocessing_beta/data_root/dataset_15_18_original/metadata_original.csv'
dset_path = '/home/jenish/dfproj/preprocessing_beta/data_root/dataset_15_18_original'

data = read_csv(filename)
real = data[0]
fake = fake_list_filter(data[0], data[1])
aggregated_list = real + fake
random.shuffle(aggregated_list)
print(len(aggregated_list))
vidnames = [x[0] for x in aggregated_list]


with open(f'{dset_path}/metadata_filtered.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for elem in aggregated_list:
        writer.writerow(elem)


vid112_dir = os.path.join(dset_path, 'videos_112') 
vid224_dir = os.path.join(dset_path, 'videos_224') 
vid112_list = glob.glob(vid112_dir + '/*.mp4')
vid224_list = glob.glob(vid224_dir + '/*.mp4')


for i in vid112_list:
    if i.split('/')[-1] not in vidnames:
        os.remove(i)

for i in vid224_list:
    if i.split('/')[-1] not in vidnames:
        os.remove(i)

print(len(glob.glob(vid112_dir + '/*.mp4')))
print(len(glob.glob(vid224_dir + '/*.mp4')))
