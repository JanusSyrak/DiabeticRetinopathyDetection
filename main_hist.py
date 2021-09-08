




"""
1. Remove folders 2, 3 and 4
2. Augment images, so the datasets are balanced
3. Compute histogram of train/0
4. Compute histogram of train/1
5. Plot histograms
6 Compare histograms
7. Select thresholds based on the previous comparison
"""

"""
Python: cv.CompareHist(hist1, hist2, method)
CV_COMP_CORREL Correlation
CV_COMP_CHISQR Chi-Square
CV_COMP_INTERSECT Intersection
CV_COMP_BHATTACHARYYA Bhattacharyya distance
CV_COMP_HELLINGER
"""