import koreanize_matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt

iris_data = sns.load_dataset('iris')
sns.pairplot(iris_data, hue='species', height=2)
plt.show()