import re
import pathlib
import tqdm

project_dir = pathlib.Path(__file__).resolve().parents[2]

print("Load linux file")
with open(f"{project_dir}\\src\\data\\posters_linux.txt") as file:
    l_lin = list(file)
    

print("Load windows file")
with open(f"{project_dir}\\src\\data\\posters_win.txt") as file:
    l_win = list(file)


print("Compare files")
missing = [ele for ele in l_lin] 
for a in tqdm.tqdm(l_win, total=len(l_win)): 
  if a in l_lin: 
    missing.remove(a) 

print(len(missing))
print("Write missing file")
with open(f"{project_dir}\\src\\data\\win_missing_posters.txt", "w") as file:
    file.writelines(missing)
