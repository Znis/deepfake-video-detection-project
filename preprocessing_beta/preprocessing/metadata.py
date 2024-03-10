from utils import get_original_with_fakes, get_originals_and_fakes
import random
import csv

#For new batch of dataset only
vid_data = get_originals_and_fakes("data_root")
b = get_original_with_fakes("data_root")
f_o = {}
for item in b:
    if item[1] in vid_data[1]:
        f_o.update({item[1]:item[0]})

wr = []
for i in vid_data[0]:
    filename = f'{i}.mp4'
    wr.append([filename,"REAL"])
for i in vid_data[1]:
    filename = f'{i}.mp4'
    wr.append([filename,"FAKE",f_o[i]])

random.shuffle(wr)
with open('metadata_test.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for elem in wr:
        writer.writerow(elem)

    
