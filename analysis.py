import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("focus_report.csv")

plt.plot(data["Time"], data["Focus %"])
plt.xlabel("Time (seconds)")
plt.ylabel("Focus %")
plt.title("Focus Over Time")

plt.show()