# EEG-SIGNAL: Implementing the research paper mentioned below using the following method:

EElectroencephalography (EEG) is an electrophysiological monitoring method to record
electrical activity of the brain. A brain–computer interface is a direct communication
pathway between any brain activity and an external device.
Using motor imagery based BCI a user can generate induced activity by imagining motor movements.
In this project, we propose to use:
DWT coefficients as features for motor imagery classification from EEG signals.
We apply Power Spectral Density over the coefficients of DWT followed by:
Principal Component Analysis for dimensionality reduction.

The other feature extraction methods of Fourier transform or sub-band energy and entropy
derived from wavelet transform eliminate the temporal information essential for the
analysis of EEG signals. However DWT coefficients contain temporal information of the
analysed signal thus it fully utilizes the simultaneous time frequency analysis preserving
the temporal information.

We perform classification of the input signal on the basis of
this analysis. We also study and compare the effect of different classifiers.
Two classifiers used are: Support Vector Machine (SVM) and k- Nearest Neighbour (k-NN).
The experiment consisted of 280 trials for 7 runs of which 140 were used for training
and 140 trials were used in testing and determining the accuracy.

Future Scope of this work:

Improvising already done work:
Work on real-time data and give real-time output.
Different method for feature extraction to improve accuracy.
Defining a new model based on the testing data to improve efficiency and accuracy.

Practical application:
Identify a person by unique brain activity. 
Password security. 
Home automation with the help of IOT. 
Learning enhancer. 
Gaming alert.
