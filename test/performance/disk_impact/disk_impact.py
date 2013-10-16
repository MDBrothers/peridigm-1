#! /usr/bin/env python

import sys
import os
import glob
import string
from subprocess import Popen, PIPE

test_dir = "disk_impact"
base_name = "disk_impact"

def read_line(file):
    """Scans the input file and ignores lines starting with a '#' or '\n'."""
    
    buff = file.readline()
    if len(buff) == 0: return None
    while buff[0] == '#' or buff[0] == '\n':
        buff = file.readline()
        if len(buff) == 0: return None
    return buff

if __name__ == "__main__":

    result = 0

    # change to the specified test directory
    os.chdir(test_dir)

    # open log file
    log_file_name = base_name + ".log"
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    logfile = open(log_file_name, 'w')

    # log file will be dumped if verbose option is given
    verbose = False
    if "-verbose" in sys.argv:
        verbose = True

    machine_name = "None"
    if "-machine" in sys.argv:
        machine_name = sys.argv[sys.argv.index("-machine") + 1]
    else:
        logfile.write("\n**** Error, machine name argument required (-machine my_machine_name)\n")
        result = 1

    # remove old output files, if any
    files_to_remove = glob.glob(base_name+".e*")
    files_to_remove.append('*.out')
    files_to_remove.append('*.nem')
    files_to_remove.append('*.pex')
    for file in os.listdir(os.getcwd()):
      if file in files_to_remove:
        os.remove(file)

    # decompose the mesh file
    command = ["../../../scripts/decomp", "-p", "4", base_name+".g"]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    if return_code != 0:
        result = return_code
    logfile.flush()
    
    # run Peridigm
    command = ["mpiexec", "-np", "4", "../../../src/Peridigm", base_name+".xml"]    
    p = Popen(command, stdout=PIPE)
    return_code = p.wait()
    if return_code != 0:
        result = return_code
    out, err = p.communicate()
    if out != None:
        logfile.write(out)
    if err != None:
        logfile.write(err)
    logfile.flush()
        
    # concatenate output files
    command = ["../../../scripts/epu", "-p", "4", base_name]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    if return_code != 0:
        result = return_code

    # compare performance statistics against gold statistics

    # performance data for current run
    stdout_vals = string.splitfields(out)
    wallclock_time_index = stdout_vals.index("Total")
    wallclock_time = float(stdout_vals[wallclock_time_index+2])

    # gold standard performance data for this machine
    perf_gold_file = open(base_name+".perf")
    buff = read_line(perf_gold_file)
    gold_perf_data = []
    while buff != None:
        vals = string.splitfields(buff)
        if machine_name in vals:
            gold_perf_data = vals
        buff = read_line(perf_gold_file)
    if gold_perf_data == []:
        logfile.write("\n**** Error, reference (gold) performance data not found for machine " + machine_name + "\n")
        result = 1
    gold_num_proc = int(gold_perf_data[1])
    gold_wallclock_time = float(gold_perf_data[2])
    gold_wallclock_time_tolerance = float(gold_perf_data[3])

    if(wallclock_time > gold_wallclock_time + gold_wallclock_time_tolerance):
        result = 1
        logfile.write("\n**** PERFORMANCE TEST FAILED:  wallclock time exceeded benchmark value plus tolerance.")
    elif(wallclock_time < gold_wallclock_time - gold_wallclock_time_tolerance):
        result = 1
        logfile.write("\n**** PERFORMANCE TEST FAILED:  wallclock time was LESS than benchmark value minus tolerance (code is running TOO FAST!).")
    else:
        logfile.write("\n**** PERFORMANCE TEST PASSED:  wallclock time was within tolerance.")
    logfile.write("\n****                           wallclock time  = " +  str(wallclock_time))
    logfile.write("\n****                           benchmark value = " +  str(gold_wallclock_time))
    logfile.write("\n****                           tolerance       = " +  str(gold_wallclock_time_tolerance) +"\n")
    logfile.flush()
          
    # compare output against gold file only if the gold file is present
    gold_file_name = base_name + "_gold.e"
    if os.path.exists(gold_file_name):
        command = ["../../../scripts/exodiff", \
                   "-stat", \
                   "-f", \
                   base_name+".comp", \
                   base_name+".e", \
                   base_name+"_gold.e"]
        p = Popen(command, stdout=logfile, stderr=logfile)
        return_code = p.wait()
        if return_code != 0:
            result = return_code
    else:
        logfile.write("\n**** Gold file " + gold_file_name + " not found, skipping exodiff.\n\n")

    logfile.close()

    # dump the output if the user requested verbose
    if verbose == True:
        os.system("cat " + log_file_name)

    sys.exit(result)
