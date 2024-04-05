import os
import subprocess
import multiprocessing
import pandas as pd
import numpy as np
import statistics
from datetime import datetime, timedelta
import sys
import shutil
from zipfile import ZipFile
from Puffball import Problem
from bayes_opt import BayesianOptimization
import argparse
import itertools
import platform
import time

try:
  import clr
  sys.path.append(r"C:\LogAnalyzer2\bin")
  clr.AddReference(r"Intel.LogAnalyzer.Utility")
  from Intel.LogAnalyzer import Utility
except:
  pass

def myNew() -> None:
   print('I am done')
   print('I am now done')

def log(inp_str):
  """
  Logs relevant outputs to weightsAdjust_log.txt
  Also prints output to console for debugging
  """
  write_str = str(inp_str)
  print(write_str)
  f.write(f"{datetime.today().strftime('%Y-%m-%d %H:%M:%S')} | {write_str}\n")

################## Experiment Setup
## Given a ton of WIP archive files (1 per every 3 minutes), get the ones we need according to the logic 

def experiment_setup(cluster, CONFIG, SITE, WIP_SCENARIO_FOLDER, EXPERIMENT_START_DATE, FILTERED_SCENARIO_FOLDER):
  """
  Does required experiment initialization
    1. Checks needed folders; if they don't exist, create them
    2. Get desired WIP files from set of all WIP files based on defined criteria (25 samples over 7 days * 2, one starting at 7:30am and one at 7:30pm)
    3. Move desired WIP files (zipped) to FILTERED_SCENARIO_FOLDER
    4. From each zip file, grab the file we want, rename it, and move it to .\ILPClusterScheduler\Output\<cluster>\<timestamp>
  """
  def check_folders():
    """
    Checks required folders to run ILP autoweights adjustment.
    If a required directory doesn't exist, it will be created.
    
    Needed folders for the ILP autoweights adjustment:
      1) .\wt_backups
        1a) .\wt_backups\<clustername> (per cluster)
      2) .\scenario_backups
      3) .\ILPClusterScheduler
        3a) .\ILPClusterScheduler\Output
          3ai) .\ILPClusterScheduler\Output\<clustername> (per cluster)
            3aia) .\ILPClusterScheduler\Output\<clustername>\Metrics (per cluster)
        3b) .\ILPClusterScheduler\<clustername> (per cluster)
      4) .\experiment_outputs
        4a) .\experiment_outputs\<clustername> (per cluster)
          4ai) .\experiment_outputs\<clustername>\all_metrics (per cluster)
    """
    required_folders_nocluster = [
      Rf".\ILPClusterScheduler",
      Rf".\ILPClusterScheduler\Output",
      Rf".\experiment_outputs",      
      Rf".\ILPClusterScheduler\Output\{cluster}\Multiweight",
      Rf".\ILPClusterScheduler\Output\{cluster}\Singleweight",
      Rf".\weight_file_backups"      
      ]

    cluster_folders = [
      Rf".\ILPClusterScheduler\Output\{cluster}",
      Rf".\ILPClusterScheduler\Output\{cluster}\Metrics",
      Rf".\ILPClusterScheduler\{cluster}",
      Rf".\experiment_outputs\{cluster}",
      Rf".\WIPScenarios\{cluster}",
      Rf".\experiment_outputs\{cluster}\all_metrics",
      Rf".\weight_file_backups\{cluster}"
      ]

    stop_condition = 0
    for required_folder in required_folders_nocluster + cluster_folders:
      if os.path.exists(required_folder):
        if os.path.isdir(required_folder):
          log(f"Directory already created at {required_folder}")
      else:
        log(f"Could not find {required_folder}. Making it.")
        os.makedirs(required_folder)
        if required_folder == Rf".\ILPClusterScheduler\{cluster}":
          log(f"WARNING: You will still need to paste the data and scheduler files to {required_folder}. You will also need to configure the scheduler.")
          stop_condition = 1

    return stop_condition

  def build_active_node_paths():
    """
    NOT used in dev environment. Uses LA2's C# Utility to enumerate active nodes.
    
    Inputs:
      - domain (str): domain to search (i.e. "RF3PROD")
      - nodes (str): str of one node. To query multiple nodes use wildcard (i.e. "RF3SAP211N*)
    Outputs
      - paths (list of strs): path to <node>/d$/LOGS/OEM for each active node
    """
    #log("BEGIN: build_active_nodes "+datetime.now().strftime('%d/%m/%Y %H:%M:%S'))      
    if CONFIG == 'STG' or CONFIG == 'PRD':
      if CONFIG == 'STG':
        domain = f'{SITE}STG'
        nodes = f'{SITE}SVAP116*'
      elif CONFIG == 'PRD':
        domain = f'{SITE}PROD'
        nodes = f'{SITE}PVAP116*'          
      try:      
        #Execute GetNodeList from LA2 c# API 
        nodelist_active = Utility.UtilityMethods.GetNodeList(domain, nodes, "")
        #log("SUCCESS: build_active_nodes "+datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
      except:
        log("ERROR: build_active_nodes "+datetime.now().strftime('%d/%m/%Y %H:%M:%S'))  
        return "NULL"
    if CONFIG == 'DEV':
      nodelist_active = [platform.node().upper()]
    return [x.lower() for x in nodelist_active]

  def get_all_wip_files(nodes):
    """
    Returns
    -----------
    all_wip_files (list): List of all .zip WIP scenario files. In a dev environment this is based on WIP_SCENARIO_FOLDER.
    - Each element contains [node (str), filename (str), filename_as_datetime (datetime)]
    
    Enumerates all WIP files and converts file suffix to datetime object. All relevant information is stored in a element.
    - We need a separate function because WIP files may be stored on multiple nodes. The list above is used to keep track of which node a file is on as well.
    - In a dev environment, node = '' 
    """
    all_wip_files = []
    if CONFIG == "STG" or CONFIG == "PRD":
      for node in nodes:
        wip_scenario_folder_node = rf"\\{node}\D$\Logs\ILP\{cluster}"
        for fpath in os.listdir(wip_scenario_folder_node):
          if "_CS_" in fpath or "_GEN_" in fpath:
            all_wip_files.append([node, fpath])
    else:
        for fpath in os.listdir(WIP_SCENARIO_FOLDER):
          if "_CS_" in fpath or "_GEN_" in fpath or "Plan_" in fpath:
            all_wip_files.append(['', fpath])
    log(f"Total wip files: {len(all_wip_files)}")
    all_wip_files_nodupes = [i for n, i in enumerate(all_wip_files) if i not in all_wip_files[1][:n]] 
    log(f"Total wip files WITHOUT DUPLICATES: {len(all_wip_files_nodupes)}")
    for fname in all_wip_files:
      if ".zip" in fname[1]:
        fname_list = [int(x) for x in fname[1].split(".zip")[0].split("_")[-5:]]
        dt_obj = datetime(*fname_list)
        fname.append(dt_obj)
    all_wip_files.sort(key=lambda x: x[2])

    return all_wip_files

  def get_relevant_wip_files(wip_files, forward_hr, num_samples=25, total_days=7, tol_mins=10):
    """
    Returns
    -----------
    files_to_move (list): subset of the wip_files passed to the function that contains only the ones we run the scheduler for.
    
    Goes through all WIP files (result of get_all_wip_files()) and gets only the ones we want according to the criteria below:
      - 26*2 WIP files over 7 days where each is equal time apart
          a) One pass starts at 7:30AM today for shift 1 (26 files) and ends at 7:30AM 7 days ago
          b) One pass starts at 7:30PM YESTERDAY for shift 3 (26 files) and ends at 7:30PM 8 days ago

    There will not be a WIP scenario for the exact date/time we want, so we check within tol_mins (default=10) given the scheduler runs every 3 minutes.
      - If we still can't find a file within those 10 minutes, we try the whole thing again with a 30 minute tolerance

      - TO DO (MAK): Don't restart the whole thing, just look for the one file
    """
    start_forward_hr = forward_hr
    files_to_move = []
    missed_dts = []
    sample_spacing = total_days / num_samples
    # Go back one day because 19:30 start won't have any data for the current day
    log(forward_hr)
    for sample in np.arange(num_samples + 1):
        # log(forward_hr)
        front_found = 0
        # Search the dates for files that we found
        for wip_list in wip_files:
            fname_datetime = wip_list[2]
            if front_found != 1:
                if (forward_hr - timedelta(minutes=tol_mins) < fname_datetime < forward_hr + timedelta(minutes=tol_mins)):
                    forward_file_str = f"{cluster}Plan_GEN_{datetime.strftime(fname_datetime, '%Y_%m_%d_%H_%M')}.zip"
                    for wip_list in wip_files:
                      if forward_file_str in wip_list[1]:
                          front_found = 1
                          files_to_move.append(wip_list)
                    else:
                        forward_file_str = f"{cluster}Plan_CS_{datetime.strftime(fname_datetime, '%Y_%m_%d_%H_%M')}.zip"
                        for wip_list in wip_files:
                          if forward_file_str in wip_list[1]:
                              front_found = 1
                              files_to_move.append(wip_list)
        if front_found == 0:
            missed_dts.append(forward_hr)
            log(f"miss for {forward_hr}")
        forward_hr -= timedelta(days=sample_spacing)

    log(f"Missed {len(missed_dts)} files")
    if len(missed_dts) > 0 and tol_mins < 30:
        log("Doing second pass with 30 min tolerance")
        return get_relevant_wip_files(wip_files, start_forward_hr, num_samples=25, total_days=7, tol_mins=30)

    return files_to_move

  def move_files(files_to_move):
      """
      Moves the desired files to FILTERED_SCENARIO_FOLDER, where the scheduler will later pick them up
      """
      log(f"Found {len(files_to_move)} files to move.")
      if len(files_to_move) < 20:
          log('Not enough files to move, aborting')
          return

      if not os.path.exists(FILTERED_SCENARIO_FOLDER):
          os.makedirs(FILTERED_SCENARIO_FOLDER)
          log(f"Made new directory: {FILTERED_SCENARIO_FOLDER}")

      curr_files = os.listdir(FILTERED_SCENARIO_FOLDER)
      if len(curr_files) > 0:
          log(f"Purging {FILTERED_SCENARIO_FOLDER}")
          for f in curr_files:
              full_file_path = Rf"{FILTERED_SCENARIO_FOLDER}\{f}"
              os.remove(full_file_path)
          log(f"Files after purging: {len(os.listdir(FILTERED_SCENARIO_FOLDER))}")

      for wip_list in files_to_move:
          # log(file_name)
          if CONFIG == 'STG' or CONFIG == 'PRD':
            origin_file = Rf"\\{wip_list[0]}\D$\Logs\ILP\{cluster}\{wip_list[1]}"
          elif CONFIG == 'DEV':
            origin_file = Rf"{WIP_SCENARIO_FOLDER}\{wip_list[1]}"              
          if os.path.exists(origin_file):
              dest_file = Rf"{FILTERED_SCENARIO_FOLDER}\{wip_list[1]}"
              shutil.copy2(origin_file, dest_file.replace("_GEN", "").replace("_CS", ""))
          else:
              log(f"Issue with {origin_file}. Please Investigate")
      log("Moved all files.")
      return None

  def unzip_moved_files():
    """
    From each .zip scenario file, we need only 1 csv input. This function extracts that input to input_folder and renames it accordingly
    """
    scheduler_paths = []
    files = os.listdir(FILTERED_SCENARIO_FOLDER)
    for file_n in files:
      if ".zip" in file_n:
        filename = file_n.split(".zip")[0]
        full_filename = Rf"{FILTERED_SCENARIO_FOLDER}\{file_n}"
        input_folder = Rf".\ILPClusterScheduler\Output\{cluster}\Multiweight\{filename}"
        scheduler_paths.append(input_folder)
        try:
          with ZipFile(full_filename, "r") as zipObj:
            zipObj.extract(Rf"{cluster}_PLANNER_INPUT.CSV", path=input_folder)
        except:
          return "ERROR"
        input_file = Rf"{input_folder}\{cluster}_PLANNER_INPUT.CSV"
        new_input_filename = Rf"{input_folder}\{filename}_INPUT.csv"
        if os.path.exists(new_input_filename):
          os.remove(new_input_filename)
        os.rename(input_file, new_input_filename)
    return scheduler_paths

  if check_folders() == 1:
    log('Cannot start because scheduler folder does not exist. Please move it.')
    return 
  # Shift 1 Starts at 7:30AM. We keep today because we'll have data for today if the script runs after 7:30AM PST
  shift1_start_time = EXPERIMENT_START_DATE.replace(hour=7,minute=30,second=0,microsecond=0)
  # Shift 3 starts at 7:30PM. We need to go back to yesterday to start because we won't have data for 7:30PM today
  shift3_start_time = (EXPERIMENT_START_DATE - timedelta(days=1)).replace(hour=19,minute=30,second=0,microsecond=0)
  # Get ALL WIP files
  nodes = build_active_node_paths()
  all_wip_files = get_all_wip_files(nodes) 
  relevant_wip_files_shift1 = get_relevant_wip_files(all_wip_files, shift1_start_time)
  relevant_wip_files_shift3 = get_relevant_wip_files(all_wip_files, shift3_start_time)
  all_experiment_wip_files = relevant_wip_files_shift1 + relevant_wip_files_shift3
  if len(all_experiment_wip_files) == 0:
    log(f"No WIP files to move. Move them to WIP_SCENARIO_FOLDER.")
    return

  move_files(all_experiment_wip_files)
  scheduler_paths = unzip_moved_files()
  log(f"Passing {len(scheduler_paths)} Files to scheduler.")
  
  return scheduler_paths, nodes

def run_single_scheduler(output_path_n, cluster):
  try:
    scheduler_exe_location = Rf"{os.getcwd()}\ILPClusterScheduler\{cluster}\Scheduler\ILPClusterScheduler.exe"
    experiment_wt_path = Rf"{os.getcwd()}\ILPClusterScheduler\{cluster}\DataFeed\WeightsData.csv"  
    
    input_filename = [x for x in os.listdir(output_path_n) if 'INPUT.csv' in x][0]
    input_filename = Rf"{output_path_n}\{input_filename}"
    run_scheduler_str = Rf"{scheduler_exe_location} -i {input_filename} -o {output_path_n} -w {experiment_wt_path}"
    subprocess.Popen(run_scheduler_str, shell=False,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT).wait()    
    
    # The scheduler will create an output file F_IP_CS_BARC_OUTPUT
    # It will also create an input file ILP_SCHEDULER_INPUT_<cluster>.txt
    output_filename = Rf"{output_path_n}\F_IP_CS_BARC_OUTPUT.txt"
    created_input_filename = Rf"{output_path_n}\ILP_SCHEDULER_INPUT_{cluster}.txt"
    os.remove(created_input_filename)
    # No more specific filenames -- can add this in the future if needed for validation
    output_filename_final = input_filename.replace("INPUT", f"OUTPUT").replace(".csv", ".txt")
    # If file, exists already, remove it
    if os.path.exists(output_filename_final):
      os.remove(output_filename_final)
    os.rename(output_filename, output_filename_final)
  except:
    log(f"Couldn't run scheduler for {output_path_n}")
      
  try:
    schedule = pd.read_csv(output_filename_final, sep="\t")
    req = schedule[[
      "LOT","ENTITY", "PLANNED_SEQ", "CASCADE_NAME","CASCADE_ORDER","DATE_ENTERED_OPER","FIRST_WFR_START_EST",
      "LAST_WFR_END_EST","IN_MOQ","LOT_PRIORITY","AUTOSTRING","PRODUCT"]]

    req = req[req["ENTITY"] != "NoEntity"]
    req = req[req["IN_MOQ"] != "Y"]
    req = req[req["PRODUCT"] != "NOPROD   0"]
    # Changing the column to datetime object and finding the cycletime
    req["DATE_ENTERED_OPER"] = pd.to_datetime(req["DATE_ENTERED_OPER"])
    req["LAST_WFR_END_EST"] = pd.to_datetime(req["LAST_WFR_END_EST"])
    req["FIRST_WFR_START_EST"] = pd.to_datetime(req["FIRST_WFR_START_EST"])
    req["CYCLE"] = (req["LAST_WFR_END_EST"] - req["DATE_ENTERED_OPER"]) / np.timedelta64(1, "m")
    # Finding the make-span time
    make_span_time = (req["LAST_WFR_END_EST"].max() - req["FIRST_WFR_START_EST"].min()) / np.timedelta64(1, "m")
    # Sum,mean and std of cycle time
    sum_cycletime = req["CYCLE"].sum()
    mean_cycletime = req["CYCLE"].mean()
    std_cycletime = req["CYCLE"].std()
    mx_cascade_order = req["CASCADE_ORDER"].max()
    # Calculating the automon cycle time
    non_automon_cycletime = req[req["AUTOSTRING"].isna()]["CYCLE"].sum()
    automon_cycletime = req[req["AUTOSTRING"].notnull()]["CYCLE"].sum()
    # Calculating the Hotbox cycle time
    hotbox_time = req[(req["LOT_PRIORITY"] > 0) & (req["LOT_PRIORITY"] <= 3)]["CYCLE"].sum()
    non_hotbox_time = req[~((req["LOT_PRIORITY"] > 0) & (req["LOT_PRIORITY"] <= 3))]["CYCLE"].sum()

    metric_dict = {
      "STANDARD_DEVIATION_CYCLE_TIME": std_cycletime,
      "MEAN_CYCLE_TIME": mean_cycletime,
      "HOT_BOX_TIME": hotbox_time,
      "NON_HOT_BOX_TIME": non_hotbox_time,
      "AUTOMON_TIME": automon_cycletime,
      "NONAUTOMON_TIME": non_automon_cycletime,
      "TOTAL_CYCLE_TIME": sum_cycletime,
      "MAKE_SPAN": make_span_time,
    }
    return metric_dict
  except:
    log(f"Couldn't calculate metrics for {output_path_n}")
    return

def remove_outliers(df, colname):
    """
    Given df and the name of a column in that df, remove outliers based on the IQR
    - Values that are not between the 1st and 3rd quartile are removed
    """
    log(f"Rows before removing outliers: {len(df)}")
    # Calculate IQR
    q3, q1 = np.percentile(df[colname], [75, 25])
    iqr_range = q3 - q1

    # Use 2 * IQR because 1.5 resulted in too many outliers
    max_val = q3 + 2*iqr_range
    min_val = q3 - 2*iqr_range

    # log(f"Min Threshold for Outliers: {min_val}")
    # log(f"Max Threshold for Outliers: {max_val}")

    n_low_values = len(df.loc[df[colname] < min_val,colname])
    n_high_values = len(df.loc[df[colname] > max_val,colname])

    log(f"Rows to drop: {n_low_values + n_high_values}")
    df_filtered = df[(df[colname] < max_val) & (df[colname] > min_val)]
    log(f"Length of Final Df: {len(df_filtered)}")

    return df_filtered

def calculate_overall_metrics(all_metrics, cluster):
  """
  Given a df with results of all 52 scheduler runs, remove outliers on desired columns and calculate our evalMetric
  - The evalMetric is essentially a proxy for cycle time, so by minimizing this value we minimize cycle time
  - The evalMetric = Mean(Mean_CT) + 2 * st.dev(Mean_CT), where Mean_CT is a vector of 52 cycle times, 1 for each WIP scenario
  """
  ratio_metric = pd.DataFrame.from_dict(all_metrics)
  ratio_metric_na = ratio_metric.copy()
  log(f"Rows in ratio_metric: {len(ratio_metric_na)}")  
  ratio_metric = ratio_metric_na.dropna(inplace=False, subset=['MEAN_CYCLE_TIME', 'STANDARD_DEVIATION_CYCLE_TIME'])
  log(f"Null rows dropped: {len(ratio_metric_na) - len(ratio_metric)}")
  if len(ratio_metric) == 0:
    log('Metric data empty after filtering')
    return "ERROR"
  ratio_metric = remove_outliers(ratio_metric, 'MEAN_CYCLE_TIME')
  ratio_metric = remove_outliers(ratio_metric, 'STANDARD_DEVIATION_CYCLE_TIME')
  if len(ratio_metric) == 0:
    log('Metric data empty after filtering')
    return "ERROR"
  metric_dir = Rf".\ILPClusterScheduler\Output\{cluster}\Metrics"
  if not os.path.exists(metric_dir):
    os.makedirs(metric_dir)
  metric_str = "Metrics"
  # for i in weights.values():
  #   metric_str += f"_{i}"
  #metric_path = Rf"{metric_dir}\{cluster}_Metrics_{metric_str}.csv"
  #ratio_metric.to_csv(metric_path, index=False)
  eval_metric = ratio_metric["MEAN_CYCLE_TIME"].mean() + 3 * statistics.stdev(ratio_metric["MEAN_CYCLE_TIME"])
  # Gather Metrics
  # means = ratio_metric.mean().add_suffix("_mean")
  # stds = ratio_metric.std().add_suffix("_std")
  # outputs = pd.DataFrame(pd.concat([means, stds]))
  # log(f"Metric Value: {eval_metric}")
  return eval_metric

def change_weights(weights, wt_path, prod=False):
  
  weight_df = pd.read_csv(wt_path)

  # If Prod is true, we want to back up the weights
  if prod == True:
    try:
        shutil.copy2(wt_path, Rf".\weight_file_backups\{cluster}\WeightsData_{datetime.today().strftime('%m-%d-%Y')}.csv")
    except Exception as ex:
        log(f"WARNING: Issue backing up weights file from {wt_path}. Sleeping for 2 minutes and trying again.")
        log(ex)
        time.sleep(120)
        try:
            shutil.copy2(wt_path, Rf".\weight_file_backups\{cluster}\WeightsData_{datetime.today().strftime('%m-%d-%Y')}.csv")
        except Exception as ex:
            log(f"Could not back up weights file from {wt_path} after retrying.")
            log(ex)

  for param_n, weight_n in weights.items():
    # If Prod is true and weight is within criteria
    # 1. Can't move weight more than 0.2
    # 2. Can't move weight if it's out of [0,1] bound
    if prod == True: 
        # Store current weight for comparison
        curr_wt = weight_df.loc[weight_df['PARAMETER_NAME'] == param_n, "VALUE"].values[0]
        if (abs((curr_wt*100 - weight_n*100)/100) >= 0.20 or curr_wt < 0 or curr_wt > 1):
            print(f"Delta for {param_n} too high or out of bounds. Current weight was {curr_wt} and new weight is {weight_n}.")
            continue
        else:
            weight_df.loc[weight_df['PARAMETER_NAME'] == param_n, "VALUE"] = weight_n
            weight_df.loc[weight_df['PARAMETER_NAME'] == param_n, "UPDATE_TIME"] = datetime.today().strftime('%m/%d/%Y %H:%M')
            weight_df.to_csv(wt_path, index=False)
    
    # Prod is false -- we just always change the weights
    else:
        weight_df.loc[weight_df['PARAMETER_NAME'] == param_n, "VALUE"] = weight_n
        weight_df.to_csv(wt_path, index=False)            
  
  return weight_df

################## Black Box Function
## The function will take in a dictionary of weights and output a number that represents cycle time
## Each time weights are passed, the scheduler runs at most 52 times (1 per WIP scenario) with those weights

def test(**weights):
  # parent_path = Rf'{os.getcwd()}\ILPClusterScheduler\Output\{cluster}\Multiweight'
  # all_output_dirs = [parent_path + '\\' + x for x in os.listdir(parent_path)]
  change_weights(weights, experiment_wt_path)

  all_metrics = []
  with multiprocessing.Pool(PROCESSES) as pool:
    for metric_dict in pool.starmap(run_single_scheduler,[(i, cluster) for i in all_experiment_wip_files]):
      if metric_dict:        
        all_metrics.append(metric_dict)
  eval_metric = calculate_overall_metrics(all_metrics, cluster)
  return eval_metric * -1

################## Optimization Functions
# Basically a wrapper for test() which takes in weights and spits out cycle time

def optimize_exhaustive():
  """
  Example of exhaustive search where we run 10 iterations varying the parameter at 0.1 increments from 0 to 1
  - Not really exhaustive with 10 iterations but could be extended easily
  """
  start_time = datetime.today()
  num_elements = len(PARAMS)
  start = [0] * num_elements
  end = [1] * num_elements

  res = []
  combinations = itertools.product(range(11), repeat=num_elements)
  for c in combinations:
    weights = [i/10 for i in c]
    weights = [start[i] + weights[i] for i in range(num_elements)]
    if any([weights[i] > end[i] for i in range(num_elements)]):
      continue
    
    weight_dict = {PARAMS[i]:weights[i] for i in range(len(PARAMS))}
    experiment_result = test(**weight_dict) * -1
    log(f"{weight_dict} --> {experiment_result}")
    
    output_dict = {PARAMS[i]:weights[i] for i in range(len(PARAMS))}
    output_dict['evalMetric'] = experiment_result
    res.append(output_dict.copy())

  # Save results to df if we want to write them later
  res_df = pd.DataFrame.from_dict(res)
  end_time = datetime.today()
  date_str = end_time.strftime('%Y-%m-%d')
  log('RUNTIME EXHAUSTIVE: ' + str(end_time - start_time))   
  res_df = res_df.reset_index().rename(columns={'index': 'iteration'}) 
  res_df.to_csv(Rf".\experiment_outputs\{cluster}\exhaustive_experiment_{date_str}.csv", index=False)
  best_iter_row = res_df.iloc[res_df['evalMetric'].idxmin()]
  log(best_iter_row)
  # log(f"BEST Val = {best_iter_row['val']} -- EvalMetric = {best_iter_row['evalMetric']}")
  return res_df

def optimize_puffball(n_iter):
  start_time = datetime.today()
  i = 0
  res = []

  def f2(x):
    nonlocal i
    i += 1
    input_dict = {PARAMS[i]:x[i] for i in range(len(PARAMS))}
    metric = test(**input_dict) * -1
    metric = round(metric, 2)
    log(f"\n{i}: {x} ---> {metric}\n")

    output_dict = input_dict.copy()
    output_dict['evalMetric'] = metric
    res.append(output_dict)
    return metric

  problem = Problem(bounds=[0,1]*len(PARAMS), target=f2)
  # Cross shows how far we want to test (+/- 1 tests the center and all the sides)
  result = problem.Minimize(stop_tests=n_iter, cross=0.0, local_vs_global=1)
  log(result)  
  df_res = pd.DataFrame.from_dict(res).reset_index().rename(columns={'index':'iteration'})
  end_time = datetime.today()
  date_str = end_time.strftime('%Y-%m-%d')
  log('RUNTIME PUFFBALL: ' + str(end_time - start_time))  
  df_res.to_csv(Rf".\experiment_outputs\{cluster}\puffball_experiment_{date_str}.csv", index=False)
  df_res = df_res.set_index('iteration')
  best_wt_dict = df_res[df_res['evalMetric'] == df_res['evalMetric'].min()].to_dict(orient='records')[0]
  best_wt_dict.pop('evalMetric')
  return best_wt_dict
  
def optimize_bayesian(init_points, n_iter):
  """
  Optimize the same thing as optimize_exhaustive(), but use BayesianOptimization package/algorithm
  - Works by initializing a few points and approximating the posterior that resembles it
  - As it runs, it gets more points and better knows where in the space to explore
  """
  start_time = datetime.today()
  # To run multiple iterations with BayesianOptimization
  pbounds = {}
  for i in range(len(PARAMS)):
    pbounds[PARAMS[i]] = (0,1)

  optimizer = BayesianOptimization(
    f=test,
    pbounds=pbounds,
    random_state=7,
    allow_duplicate_points=True
  )
  optimizer.maximize(
    init_points=init_points,
    n_iter=n_iter
    )
  log(f"Best: {optimizer.max}")
  # Turn results into dataframe for further analysis
  df = pd.DataFrame(optimizer.res)
  df_params = df['params'].apply(lambda x: pd.Series(x))
  df_res = pd.concat([df['target'], df_params], axis=1)
  df_res = df_res[PARAMS + ['target']]
  df_res = df_res.reset_index().rename(columns={'index': 'iteration', 'target': 'evalMetric'})
  #df_res = df_res.round(2)
  end_time = datetime.today()
  date_str = end_time.strftime('%Y-%m-%d')
  log('RUNTIME BAYESIAN: ' + str(end_time - start_time))    
  df_res.to_csv(Rf".\experiment_outputs\{cluster}\bayesian_experiment_{date_str}.csv", index=False)
  return df_res

def optimize_local_exhaustive(iter_cntr):
  
  start_time = datetime.today()
  weight_dict = dict.fromkeys(PARAMS)
  df = pd.read_csv(PROD_WT_SRC)

  for param in PARAMS:
    val = int(df[df['PARAMETER_NAME'] == param]['VALUE'].values[0]*100)
    weight_dict[param] = val 
  log(f"INPUT PROD WEIGHTS: {weight_dict}")  
  
  res = []
  weight_dict_experiment = weight_dict.copy()
  for i in range(weight_dict[PARAMS[0]]-iter_cntr, weight_dict[PARAMS[0]]+iter_cntr+1, 1):
    weight_dict_experiment[PARAMS[0]] = i/100
    for j in range(weight_dict[PARAMS[1]]-iter_cntr, weight_dict[PARAMS[1]]+iter_cntr+1, 1):
      weight_dict_experiment[PARAMS[1]] = j/100
      experiment_result = test(**weight_dict_experiment)
      log(f"{weight_dict_experiment} --> {experiment_result}")

      output_dict = weight_dict_experiment.copy()
      output_dict['evalMetric'] = experiment_result
      res.append(output_dict)
      
      #Add another loop for third param
      if len(PARAMS) == 3:
        for k in range(weight_dict[PARAMS[2]]-iter_cntr, weight_dict[PARAMS[2]]+iter_cntr+1, 1):
          weight_dict_experiment[PARAMS[2]] = k/100

  res_df = pd.DataFrame.from_dict(res).reset_index().rename(columns={'index':'iteration'})
  end_time = datetime.today()
  date_str = end_time.strftime('%Y-%m-%d')
  log('RUNTIME LOCAL EXHAUSTIVE: ' + str(end_time - start_time))     
  res_df.to_csv(Rf".\experiment_outputs\{cluster}\local_exhaustive_experiment_{date_str}.csv", index=False)
  return res_df

# User Inputs
parser = argparse.ArgumentParser(description='Config for ILP Automatic Weights Adjustment.')
parser.add_argument('clusters')
parser.add_argument('config')
parser.add_argument('site')
parser.add_argument('methods')
parser.add_argument('--change_weight_param', action="store_true", default=False)


args = parser.parse_args()
clusters = [x.upper() for x in args.clusters.split('::')]
config = args.config.upper()
site = args.site.upper()
methods = [x.upper() for x in args.methods.split('::')]
change_weight_param = args.change_weight_param

# Stuff we set
PARAMS = ['WaferStartPenalty', 'ProcessTimePenalty', 'CascadePenalty', 'ToolUtilizationPenalty', 'RfsPrioWeight']
N_ITER = 800
PROCESSES = int(multiprocessing.cpu_count() / 2)
start_date = datetime.today()
if config == "DEV":
  start_date = datetime(2023, 3, 27)
  N_ITER = 2
  PROCESSES = int(multiprocessing.cpu_count())
f = open('weightsAdjust_log.txt', 'a+')

############## USAGE IF RUNNING FROM THE COMMAND LINE YOURSELF
# Args (specified above) are : clusters config site methods, and they are taken with only spaces between
# You can use multiple clusters and methods separated by 2 semicolons
# Example (dev box) : TBX76::TBC72 DEV DEV EXHAUSTIVE::PUFFBALL
# Example (dev box) : TBX76 DEV DEV PUFFBALL


if __name__ == '__main__':
  for cluster in clusters:
    PROD_WT_SRC = Rf"C$\FabAuto\ILPClusterScheduler\{cluster}\DataFeed\WeightsData.csv"
    wip_scenario_folder = Rf".\WIPScenarios\{cluster}"
    filtered_scenario_folder = Rf".\ILPClusterScheduler\WIPScenarios\{cluster}\filtered"   
    experiment_wt_path = Rf"{os.getcwd()}\ILPClusterScheduler\{cluster}\DataFeed\WeightsData.csv"
    # Also grabs the active nodes because we need to iterate through them when we change the weights
    all_experiment_wip_files, active_nodes = experiment_setup(cluster, config, site, wip_scenario_folder, start_date, filtered_scenario_folder)
    if all_experiment_wip_files:
      for method in methods:
        
        METHOD_START_TIME = datetime.now()
        print(f"{method} START: {METHOD_START_TIME}")
        
        if method == "EXHAUSTIVE":
          optimizer_result = optimize_exhaustive()
        elif method == "BAYESIAN":
          optimizer_result = optimize_bayesian(init_points=10, n_iter=N_ITER-10)
        elif method == "PUFFBALL":      
          optimizer_result = optimize_puffball(n_iter=N_ITER)
        elif method == "LOCAL_EXHAUSTIVE":
          optimizer_result = optimize_local_exhaustive(iter_cntr=5)
        else:
          log('Invalid Method. Try again.')
        
        if change_weight_param is True:
            # Change weights on all active nodes
            for node in active_nodes:
                try:
                    node_path = Rf"\\{node}\{PROD_WT_SRC}"
                    change_weights(optimizer_result, node_path, prod=True)
                    log(f"Changed weights at {node_path}")
                except:
                    log(f"ERROR: Could not change weights at {node_path}. File likely in use. Sleeping for 2 minutes and retrying.")
                    try:
                        time.sleep(120)
                        change_weights(optimizer_result, node_path, prod=True)
                    except:
                        continue

        METHOD_END_TIME = datetime.now()
        print(f"{method} END: {METHOD_END_TIME}")
        print(f"TOTAL RUNTIME for {method}: {METHOD_END_TIME - METHOD_START_TIME}")

    else:
      log("ERROR: Couldn't find WIP files. Terminating.")