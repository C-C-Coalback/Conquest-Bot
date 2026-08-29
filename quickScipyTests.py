from scipy.stats import binomtest

result = binomtest(29, 40, 0.5, alternative="greater")
# p-value 0.003
print(result.pvalue)
print(result.proportion_ci())