import os
import cv2
import tqdm
import csv

source_dir = "/home/jovyan/VGGFace2/VGG-Face2/data/test/"
result_dir = "/home/jovyan/VGGFace2-aligned/test/"
bb_csv_path = "/home/jovyan/VGGFace2/VGG-Face2/meta/bb_landmark/loose_bb_test.csv"

def convert_backslashes(string: str):
    return string.replace("\\", "/")

def path_join(*paths):
    return convert_backslashes(os.path.join(*paths))

def list_all_files(path):
    paths = []
    for root, _, files in os.walk(path):
        if len(files) < 1:
            continue
        paths += [path_join(root, file) for file in files]
    return paths

def main():
    os.makedirs(result_dir, exist_ok=True)
    with open(bb_csv_path) as f:
        r = csv.reader(f, delimiter=',')
        first = True
        for row in tqdm.tqdm(r):
            if first:
                first = False
                continue
            source_filename = source_dir + row[0] + ".jpg"
            dest_path = result_dir + row[0] + ".jpg"
            img = cv2.imread(source_filename)
            X, Y, W, H = int(row[1]), int(row[2]), int(row[3]), int(row[4])
            try:
                cropped_image = img[Y:Y+H, X:X+W]
                resized_image = cv2.resize(cropped_image, (128, 128))
                os.makedirs(dest_path[:dest_path.rindex('/')], exist_ok=True)
                cv2.imwrite(dest_path, resized_image)
            except:
                continue

if __name__ == "__main__":
    main()