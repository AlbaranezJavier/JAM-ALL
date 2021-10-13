import cv2, glob, json
from multiprocessing import Pool

'''
This script pre-processes the input data to speed up the neural network training process.
'''

def task(path):
    # Image processing
    img_name = path.split("\\")[-1][:-5]
    img = cv2.imread(folder + images + img_name + ".jpg")
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, new_resolution)
    cv2.imwrite(new_destination + images + img_name + ".png", img)

    # Jsons processing
    with open(path, "r") as reader:
        data = json.load(reader)
        reader.close()
    for i in range(data["Cell Numbers"]):
        data[f"Cell_{i}"]["x1"] = str(int((int(data[f"Cell_{i}"]["x1"]) / original_resolution[0]) * new_resolution[0]))
        data[f"Cell_{i}"]["x2"] = str(int((int(data[f"Cell_{i}"]["x2"]) / original_resolution[0]) * new_resolution[0]))
        data[f"Cell_{i}"]["y1"] = str(int((int(data[f"Cell_{i}"]["y1"]) / original_resolution[1]) * new_resolution[1]))
        data[f"Cell_{i}"]["y2"] = str(int((int(data[f"Cell_{i}"]["y2"]) / original_resolution[1]) * new_resolution[1]))
    json_file = json.dumps(data, separators=(',', ':'))
    with open(new_destination + jsons + img_name + ".json", "w") as outfile:
        outfile.write(json_file)
        outfile.close()

if __name__ == '__main__':
    # original data
    folder = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\First_micrscope_all"
    images = "/images/"
    jsons = "/jsons/"
    original_resolution = (5312, 2988)
    # new data
    new_resolution = (320, 180)
    new_destination = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\Fist_microscope_all_320x180"
    # multiprocessing
    pool = Pool(10)

    # paths
    json_paths = glob.glob(folder+jsons+"*.json")
    pool.map(task, json_paths)
