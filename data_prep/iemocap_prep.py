import shutil
import pandas as pd
 
csv_file = pd.read_csv("../data/iemocap.csv")
files = csv_file["FileName"]

root = "./IEMOCAP_full_release/"
destination = "../data/iemocap/"

for file in files:
    session = file[4:5]
    session = "Session" + str(session)
    src = root + session + "/" + "/sentences/wav/" + "_".join((file.split(".")[0].split("_")[:-1])) + "/" + file + ".wav"
    dst = destination + file + ".wav"
    shutil.copyfile(src, dst)