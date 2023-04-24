import mtcnn
import os
import cv2
import tqdm

source_dir = "/home/jovyan/VGGFace2/VGG-Face2/data/test/"
result_dir = "/home/jovyan/VGGFace2-aligned/"

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

def preprocess(det, img):
    detected = det.detect_faces(img)
    print("Three")
    if detected is None or len(detected) == 0:
        return None
    if detected[0]["confidence"] < 0.7:
        return None
    bbox = detected[0]["box"]
    print("Four")
    X, Y, W, H = bbox[0], bbox[1], bbox[2], bbox[3]
    cropped_image = img[Y:Y+H, X:X+W]
    resized_image = cv2.resize(cropped_image, (128, 128))
    print("Five")
    return resized_image

def main():
    os.makedirs(result_dir, exist_ok=True)
    detector = mtcnn.MTCNN()
    filenames = list_all_files(source_dir)
    for img_path in tqdm.tqdm(filenames):
        img = cv2.imread(img_path)
        print(img.shape)
        p_img = preprocess(detector, img)
        if p_img is None:
            continue
        results_filename = img_path.replace(source_dir, result_dir)
        os.makedirs(results_filename[:results_filename.rindex('/')], exist_ok=True)
        cv2.imwrite(results_filename, p_img)
        print("Six")
        
if __name__ == "__main__":
    main()