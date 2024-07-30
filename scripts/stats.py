import numpy as np
from scipy.stats import ttest_ind, shapiro, kstest, normaltest, mannwhitneyu

def check_gaussianity(data):
    # Shapiro-Wilk Test
    stat, p = shapiro(data)
    print('Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

    # D'Agostino's K^2 Test
    stat, p = normaltest(data)
    print('D\'Agostino\'s K^2 Test: Statistics=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
    
    # Kolmogorov-Smirnov Test
    stat, p = kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    print('Kolmogorov-Smirnov Test: Statistics=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

def mannwu(group1, group2, alpha=0.05):
    # Mann-Whitney U test
    statistic, p_value = mannwhitneyu(group1, group2)
    print("Mann-Whitney U statistic: {:.2f}, p-value is {:.10f}".format(statistic, p_value))
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference in the distributions.")
    else:
        print("Fail to reject the null hypothesis. There is no significant difference in the distributions.")

def two_sample_ttest(data1, data2, alpha = 0.05, equal_var=True):
    """
    Perform a two-sample t-test on two sets of data.

    Parameters:
    - data1 (array-like): Data for sample 1.
    - data2 (array-like): Data for sample 2.
    - equal_var (bool): If True (default), perform the test assuming equal variance.

    Returns:
    - tuple: t-statistic and p-value of the t-test
    """
    # Convert data to numpy arrays for handling different types of inputs
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    # Perform the t-test
    t_stat, p_value = ttest_ind(data1, data2, equal_var=equal_var)
    if p_value < alpha:
        print("Reject the null hypothesis - suggest the sample means are different")
    else:
        print("Do not reject the null hypothesis - suggest the sample means are similar")
    return t_stat, p_value