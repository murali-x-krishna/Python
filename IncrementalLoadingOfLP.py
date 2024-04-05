import os
import subprocess
import pandas as pd
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

############################################################################
# Function to read the ng.output file and tell me
# what state it is in. E.g., Do I see a "Simulation Beginning" message but
# no "EOF" message? Or do I not see either of them? Or do I see an EOF but
# not the simulation beginning?
############################################################################
def getNGOutputState():
  # Check that ng.output file exists
  if (not os.path.isfile(NGOutput_file_withPath)):
    print(Rf"{NGOutput_file_withPath} specified does not exist")
    exit()

  # Open the ng.output file to read the lines one at a time
  ng_output_file = open(NGOutput_file_withPath, "r")

  # Reset my loop counter and condition
  loop_counter = 0
  condition = gc_continueLooping

  # Read the ng.output file until we find a line that says "Simulation Beginning"
  # If we do not find the line and find "EOF" then terminate the program.
  # If we do not find an EOF either, then wait for 60 secs, re-checking every 5 secs
  # for the "Simulation Beginning" or "EOF" to show up. When timeout reached, exit the program.
  while condition == gc_continueLooping:
    while ng_output_file:
      ng_fileLine = ng_output_file.readline()

      if (gc_simulation_beginning_str in ng_fileLine):
        condition = gc_simulation_beginning_str
        break # I.e., we can start reading LotPrediction.rep

      if (gc_EOF_str in ng_fileLine):
        condition = gc_EOF_str
        break #I.e. we reached end of the ng.output file without seeing "Simulation Beginning"
      
    # If I've reached here and the condition is still to continueLooping it means I have looped 
    # through ng.output and neither condition has triggered (I.e. Simulation beginning AND EOF not found).
    # I will sleep for 5 secs and the retry reading from the file
    if (condition == gc_continueLooping and loop_counter < 12):
      time.sleep(5) 
      loop_counter += 1
    elif (loop_counter >= 12):
      condition = gc_timeout_str
    else: 
      break # We should not get here but it is good coding practice to have a catch-all!    
  
  # Close the file
  ng_output_file.close()

  # Return the condition value that indicates the state of ng.output
  return condition
#### End of function #######################################################

############################################################################
# Function to read the ng.output file and tell me if there is an EOF. That
# will tell me that the simulation run has completed and I can stop reading
# the LotPrediction.rep
############################################################################
def isNGOutputFileEOF():
  # Check that ng.output file exists
  if (not os.path.isfile(NGOutput_file_withPath)):
    print(Rf"{NGOutput_file_withPath} specified does not exist")
    exit()

  # Open the ng.output file to read the lines one at a time
  ng_output_file = open(NGOutput_file_withPath, "r")

  # Read the ng.output file until we find a line that says "EOF"
  while ng_output_file:
    ng_fileLine = ng_output_file.readline()

    if (gc_EOF_str in ng_fileLine):
      # Close the file
      ng_output_file.close()
      return True
          
  # Close the file
  ng_output_file.close()

  # EOF not found... hence return false
  return False
#### End of function #######################################################

############################################################################
# Main part of the program
############################################################################

# User Inputs
parser = argparse.ArgumentParser(description='Read LotPrediction.rep')
parser.add_argument('LotPredictor_location_path')

# Parse args
args = parser.parse_args()
lotPredictor_location_path = args.LotPredictor_location_path.upper()

# Stuff we set based on the args
PARAMS = ['LotPredictor_location_path']
lotPredictor_file_withPath = Rf"{lotPredictor_location_path}\LotPrediction.rep"
NGOutput_file_withPath = Rf"{lotPredictor_location_path}\ng.output"

# Some constants I set
gc_simulation_beginning_str = 'Simulation Beginning'
gc_EOF_str = 'EOF'
gc_timeout_str = 'Timeout'
gc_continueLooping = 'Continue looping'
gc_minsBeforeTimeout = 3  # I will wait for an update in the ng.output file for these many mins. If none, exit the program.

# Check that LotPrediction.rep exists
if (not os.path.isfile(lotPredictor_file_withPath)):
  print(Rf"{lotPredictor_file_withPath} specified does not exist")
  exit()
  
# Get the ng.output file state
condition = getNGOutputState()
print(condition)

# Check the condition variable and decide next action
if (condition == gc_timeout_str):  # I.e. we timed out waiting for any of my desired string tokens to appear
  print(Rf"Timed out waiting for {NGOutput_file_withPath} to have a SimStart or EOF indicator")
  exit(1) 
elif (condition == gc_EOF_str):    # I.e. we reached end of file without seeing a simulation beginning
  print(Rf"Reached end of {NGOutput_file_withPath} without detecting a Simulation Beginning message")
  exit(2) 
elif (condition == gc_continueLooping):  
  print(Rf"Unhandled exception while reading {NGOutput_file_withPath}")
  exit(3) 
else:
  if (condition != gc_simulation_beginning_str):  # We should never have this condition. Specifying it anyway, just in case!
    print(Rf"Unhandled exception while reading {NGOutput_file_withPath}")
    exit(3) 

#########################################################################
# I have established at this point that the LotPrediction.rep can be read
# The code below reads from the file
#########################################################################

# Open the LotPrediction.rep file to read the lines one at a time
lp_file = open(lotPredictor_file_withPath, "r")
 
# Open temp file for writing
output_file = open(Rf"c:\temp\MK\test.tab", "w")

# Read the LotPrediction.rep header... and move to next line
fileLine = lp_file.readline()

# Reset my loop counter
loop_counter = 0

# Loop through the file reading one line at a time. Will upload that line to the DB
while lp_file:
  # Read next line
  fileLine = lp_file.readline()
  
  # print that line to the output file
  output_file.write(fileLine)

  # Flush the buffer every 100,000'th line
  if (loop_counter % 100000 == 0):
    print(loop_counter)
    output_file.flush()
  
  # If we've reached the end of the file then we need to check whether the 
  # simulation has finished. If not, will sleep and then retry reading 
  # more lines from the file
  if (fileLine == ""): 
    condition = getNGOutputState
    time.sleep(5)
    break
	
  loop_counter += 1

# Close the files - input and output
lp_file.close()
output_file.close()
print('I am done!!')
print('I thought I was done but now I really am!')