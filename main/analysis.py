import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'best_model_system_states.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path, header=None)

# Display the first few rows of the DataFrame
print(df.head())
print(df.shape)

t = np.linspace(0, 10, num=1001)

theta = df.iloc[0]
phi = df.iloc[1]

# Plot the first row against the numbers
plt.figure(figsize=(10, 6))
plt.plot(t, theta + phi)
plt.xlabel('Time (seconds)')
plt.ylabel('Robot angle (radians)')
plt.title('Robot angluar position over time for first agent to reach 5000 reward')
plt.grid(True)
plt.show()