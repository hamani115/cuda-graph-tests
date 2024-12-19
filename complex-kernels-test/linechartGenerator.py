import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('complex_3diffkernels_data.csv')
df2 = pd.read_csv('complex_3diffkernels_data.csv')

# Ensure NSTEP is sorted
df = df.sort_values('NSTEP')

# Extract NSTEP values
nsteps = df['NSTEP'].values

########################## Percentage Differences ##########################
# You can choose to plot either 'DiffPercentWithout' or 'DiffPercentWith'
percentage_difference_without = df['DiffPercentWithout'].values
percentage_difference_with = df['DiffPercentWith'].values

# Plotting
plt.figure(figsize=(10, 6))

# Plot Percentage Difference Without First Run/Graph Creation
plt.plot(nsteps, percentage_difference_without, marker='o', color='red', label='Without First Run/Graph', linestyle='-')

# Plot Percentage Difference With First Run/Graph Creation
plt.plot(nsteps, percentage_difference_with, marker='o', color='red', label='With First Run/Graph', linestyle='--')

for i, (x, y) in enumerate(zip(nsteps, percentage_difference_without)):
    plt.text(x, y, f'{y:.2f}', fontsize=10, color='black', ha='center', va='top'NVIDIA Tesla T4 (Trail 1))
    
for i, (x, y) in enumerate(zip(nsteps, percentage_difference_with)):
    plt.text(x, y, f'{y:.2f}', fontsize=10, color='black', ha='center', va='bottom')


plt.title('Total Time Difference Percentage With CUDA Graphs: 3 Different Kernels Test')
plt.xlabel('NSTEP (Number of Iterations)')
plt.ylabel('Total Difference Percentage (%)')
plt.xscale('log')  # Use log scale for NSTEP
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
# plt.show()

# Save the figure as a JPEG file
filename = 
plt.savefig('performance_improvement.jpg', format='jpg', dpi=300)



# Close the figure
plt.close()