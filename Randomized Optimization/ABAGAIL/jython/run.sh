#!/bin/bash
# edit the classpath to to the location of your ABAGAIL jar file
#
export CLASSPATH=../ABAGAIL.jar:$CLASSPATH
mkdir -p data/plot logs image

# abalone test
echo "Running abalone test"
jython abalone_test.py

echo "Running learning curve test"
jython chess_test.py

# four peaks
echo "four peaks"
jython fourpeaks.py


# continuous peaks
echo "continuous peaks"
jython continuouspeaks.py


# traveling salesman
echo "Running traveling salesman test"
jython travelingsalesman.py