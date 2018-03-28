import numpy as np

class CsvWriter:

     def write_to_csv(self, pre_z_data):
         np.savetxt("pre_z_data.csv", pre_z_data, delimiter=",", fmt='%.3f')



