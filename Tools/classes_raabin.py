import json, glob

if __name__ == '__main__':
    dir = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\first_v2_all\jsons"
    classes = []

    jsons = glob.glob(dir+"/*.json")
    total = len(jsons)
    counter = 0
    for path in jsons:
        counter += 1
        with open(path, "r") as reader:
            data = json.load(reader)
            reader.close()
        for i in range(data["Cell Numbers"]):
            for l in range(1, 3):
                if data[f"Cell_{i}"][f"Label{l}"] not in classes:
                    classes.append(data[f"Cell_{i}"][f"Label{l}"])
            print(f"{counter}/{total}", end="\r")
    print()
    print(classes)
