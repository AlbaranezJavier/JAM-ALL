import json, glob, cv2, shutil

"""
Almacena aquellos ejemplos con etiquetas en las que coinciden ambos expertos
"""

if __name__ == '__main__':
    dir = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\First_microscope_all_320x180"
    save_dir = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\first_v2_all_320x180"
    jsons = r"\jsons"
    images = r"\images"
    classes_excluded = [None, "Unknn", "Not centered", "NRBC", "Megakar"]

    paths = glob.glob(dir + jsons + "/*.json")
    total = len(paths)
    counter = 0
    for path in paths:

        with open(path, "r") as reader:
            data = json.load(reader)
            reader.close()

        save = True
        for i in range(data["Cell Numbers"]):
            if data[f"Cell_{i}"]["Label1"] != data[f"Cell_{i}"]["Label2"] \
                    or data[f"Cell_{i}"]["Label1"] in classes_excluded:
                save = False
                break

        if save:
            counter += 1
            filename = path.split("\\")[-1][:-5]
            shutil.copyfile(dir + images + f"/{filename}.png", save_dir + images + f"/{filename}.png")
            shutil.copyfile(path, save_dir + jsons + f"/{filename}.json")
            print(f"{counter}/{total}", end="\r")
