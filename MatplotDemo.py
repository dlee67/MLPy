import numpy as np
import matplotlib.pyplot as mp

#mp.scatter([1700, 2100, 1900, 1300, 1600, 2200], [53000, 65000, 59000, 41000, 50000, 68000])
#mp.plot([1700, 2100, 1900, 1300, 1600, 2200], [53000, 65000, 59000, 41000, 50000, 68000])
#mp.show()

#mp.scatter([1700, 2100, 1900, 1300, 1600, 2200], [53000, 44000, 59000, 82000, 50000, 68000])
#mp.plot([1700, 2100, 1900, 1300, 1600, 2200], [53000, 44000, 59000, 82000, 50000, 68000])
#mp.show()

#mp.scatter([1300, 1400, 1600, 1900, 2100, 2300], [88000, 72000, 94000, 86000, 112000, 98000])
#mp.plot([1300, 1400, 1600, 1900, 2100, 2300], [88000, 72000, 94000, 86000, 112000, 98000])
#mp.show()

#diff_cost = 94000-86000
#diff_size = 1900-1600

#I can't figure out how to make this thing look like the histogram from udacity: statistic 101,
#Quiz: Histograms 2.
#The frequencies appear correctly, but the representation is not intended.
salaries = np.array([132754, 137192, 122177, 147121, 143000, 126010, 129200, 124312, 128132])
mp.hist(salaries, [120000, 130000, 140000], range=None, density=None, weights=None, 
cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, 
color=None, label=None, stacked=False, normed=None, hold=None, data=None, **kwargs)
#The representations of the x-axis lables aren't how I want.
mp.show()