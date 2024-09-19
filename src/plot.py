import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set the style and font
sns.set_style("whitegrid")
sns.set(font='serif')
sns.set_style("white", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"],
})

# Optionally, increase the font size for better readability
sns.set_context("notebook", font_scale=1.5)

# Read the data from the CSV file
ds = pd.read_csv("../log_likelihood.csv")

# Ensure Severity is of integer type
ds['Severity'] = ds['Severity'].astype(int)

# Increase the plot size
plt.figure(figsize=(15, 10))  # Adjust the width and height as needed

# Create the line plot with thicker lines
ax = sns.lineplot(
    x="Severity",
    y="LogLikelihood",
    hue="Corruption",
    data=ds,
    marker='o',       # Optional: add markers to the lines
    linewidth=4.5     # Adjust the line width here
)

# Set x-axis ticks to integers from 1 to 5
plt.xticks([1, 2, 3, 4, 5])

# Move the legend outside the plot
plt.legend(
    title='Corruption Type',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0,
    ncol=2  # Adjust ncol as needed
)

# Add labels and title
plt.xlabel('Severity Level')
plt.ylabel('Average Log-Likelihood')
plt.title('Average Log-Likelihood Across Severities for All Corruptions')

# Adjust layout to accommodate the legend
plt.tight_layout()

# Save the plot, ensuring the legend is included
plt.savefig('plot.png', bbox_inches='tight')

# Display the plot
plt.show()