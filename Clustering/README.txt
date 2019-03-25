In order to run the code, you will need to use python 3.x with this code, and to pip install the packages in requirements.txt. The main addition here is the tables module which does require HDF5. If you are using OS X with Homebrew you can simply brew install hdf5 before installing the requirements. If this does not work for you, try the requirements-no-tables.txt file. Windows users have noted the need to install the tables module but on some systems this is not required.

The code is heavily based on https://github.com/cmaron/CS-7641-assignments/tree/master/assignment3

Run terminal command python run_experiment.py -h to see all your options
and run python run_experiment.py --plot to plot the results of the experiment.

The code can be found: https://github.com/monirakhan/machinelearninghw/tree/master/Clustering