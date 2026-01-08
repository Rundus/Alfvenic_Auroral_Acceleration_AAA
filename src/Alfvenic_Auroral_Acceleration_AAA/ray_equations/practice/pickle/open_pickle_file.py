# --- Practice opening a pickle file of one of my (mu,chi) functions ---

# Get the files
folder_path = r'C:\Users\cfelt\PycharmProjects\Alfvenic_Auroral_Acceleration_AAA\src\Alfvenic_Auroral_Acceleration_AAA\scale_length\pickled_expressions'
from glob import glob
pickle_files = glob(folder_path+'\\*.pkl*')

# open one
import dill

file = open(pickle_files[0],'rb')
lmb_e = dill.load(file)
print(lmb_e(-0.5, 0.1))
print(6702.360960833486)

