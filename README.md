# NEA-Patch-alignment
Written for a labmate in 2019. 

The data consists of intracellular action potentials in cardiomyocytes, recorded simultaneously through patch clamp and a custom nanoelectrode array with electrical stimulation. However, the recording duration differs for the two types of recordings, and they are also not started at the same time due to hardware limitations. However, it is possible to identify the time to synchronize the patch and nanoelectrode recordings via the sharpest rising in standard deviation of the signal (caused by the delivery of an electric stimulation pulse). 

The goal is to make a visualization tool to compare the similarity in the shape of the action potential of the patch and NEA recordings. Hence, the measurements are scaled down to the same amplitude for a visual comparison of the shape. Other information quantifying the similarity of the signals (e.g. APD, frequency, etc) are processed elsewhere.
