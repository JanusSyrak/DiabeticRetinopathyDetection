# DIARVIS
The repository for the project DIARVIS - Diabetic Retinopathy Vision

MetricsCallback.py defines the specific callback Histories, that computes sensitivity and specificity at the end of each training epoch.

ModelFactory.py presents a simplified framework for creating Deep Learning models based on the pre-trained models in Keras.

StaticDataAugmenter.py is a module for simplifying data augmentation of a data set.

auto_run.py is a module for efficiently running tests, effectively removing human interaction.

fileStream.py is a simplified interface for writing the results of the project to a file.

histogramEqualization.py contains various methods for histogram equalization, along with computing various histogram related metrics. 

main and main_hist are drivers for the various parts of the project, and can be considered obsolete.
