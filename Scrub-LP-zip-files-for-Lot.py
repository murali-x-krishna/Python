import os
import numpy as np
import statistics
from datetime import datetime, timedelta
import sys
import shutil
from zipfile import ZipFile
import argparse
import itertools
import platform
import time

# Define the log function
def log(inp_str):
  """
  Logs relevant outputs to my log file
  Also prints output to console for debugging
  """
  write_str = str(inp_str)
  f.write(f"{datetime.today().strftime('%Y-%m-%d %H:%M:%S')} | {write_str}\n")

# User Inputs
parser = argparse.ArgumentParser(description='Extract LotPredictor from archive and gather actual arrivals')
parser.add_argument('LotPredictor_archive_path')
parser.add_argument('LotPredictor_zip_archive_name')
parser.add_argument('Extracted_file_dest_path')

args = parser.parse_args()
lotPredictor_archive_path = args.LotPredictor_archive_path.upper()
lotPredictor_zip_archive_name = args.LotPredictor_zip_archive_name.upper()
extracted_file_dest_path = args.Extracted_file_dest_path.upper()

# Stuff we set based on the args
PARAMS = ['LotPredictor_archive_path', 'LotPredictor_zip_archive_name', 'Extracted_file_dest_path']
lotPredictor_zip_archive_withPath = Rf"{lotPredictor_archive_path}\{lotPredictor_zip_archive_name}"
outputFileName = Rf"LP_ActArrivals.tab"
outputFileName_withPath = Rf"{extracted_file_dest_path}\{outputFileName}"
baseDirectory = os.getcwd()

# Check that the current subdir is LotPredictorAnalysis
if (baseDirectory.split("\\")[-1] != 'LotPredictorAnalysis'):
  print('\nInvalid current location. Need to run this .py script from the LotPredictorAnalysis directory')
  exit()

# Open my log file
logFileNameWithPath = Rf"{baseDirectory}\Logs\ExtractLotPredictorNActuals.log"
f = open(logFileNameWithPath, 'a+') 

# Check that the input path and input zip archive exists 
if (not os.path.exists(lotPredictor_archive_path)):
  log('Invalid lotPredictor_archive_path specified')
  print('Invalid lotPredictor_archive_path specified')
  exit()

# Check that the output path exists 
if (not os.path.isfile(lotPredictor_zip_archive_withPath)):
  log('Invalid lotPredictor_zip_archive_name specified')
  print('Invalid lotPredictor_zip_archive_name specified')
  exit()

# Check that the output path exists 
if (not os.path.exists(extracted_file_dest_path)):
  log('Invalid extracted_file_dest_path specified')
  print('Invalid extracted_file_dest_path specified')
  exit()

log('All args are validated to be good')

# Open the zip file and extract the LotPredictor.rep file to the destination
# If LotPrediction.rep already exists on the dest location it will be overwritten
try:
  with ZipFile(lotPredictor_zip_archive_withPath, "r") as zipObj:
    zipObj.extract('LotPrediction.rep', path=extracted_file_dest_path)
except:
  log('Error unzipping file')
  print('Error unzipping file')
  exit()

# Upload the LotPrediction.rep to the MLITE PP DB
# I execute sqlldr to do this. 
os.system(Rf"sqlldr.exe control={baseDirectory}\SQL_scripts\load_NewLP_MK.ctl,log={baseDirectory}\Logs\load_NewLP_MK.log,bad={baseDirectory}\Logs\load_NewLP_MK.bad,DATA={extracted_file_dest_path}\LotPrediction.rep,userid=nexsim/4nex$im@D1D.PP.NEXSIM")

# Log progress
log('Unzipped file and extracted LotPredictor.rep')

# Download the final file with the columns we want - the LotPred columns and the actual arrival times
os.system(Rf"sqlplus.exe nexsim/4nex$im@D1D.PP.NEXSIM @{baseDirectory}\SQL_Scripts\LP_ActArrivals_MK.sql {baseDirectory}\Data\.")

# Rename the extracted file to have the datetime from the original zip archive
# First figure the new file name we want
lotPredictor_zipFileName = lotPredictor_zip_archive_name.replace(".ZIP", "")
new_outputFileName_withPath = outputFileName_withPath.replace(outputFileName, Rf"{lotPredictor_zipFileName}_{outputFileName}")

# Before renaming make sure that there is no file with the same name in the dest folder. If present, delete it.
if (os.path.exists(new_outputFileName_withPath)):
  os.remove(new_outputFileName_withPath)

# Rename the final output file
os.rename(outputFileName_withPath, new_outputFileName_withPath)

# Close the log file
f.close()
