import matplotlib.pyplot as plt
import numpy as np

# Data to plot
labels = 'Planning Vacations', 'Planning Retirement'
sizes = [55, 45]  # 55% planning vacations, 45% planning retirement
colors = ['lightblue', 'lightgreen']
explode = (0.1, 0)  # explode 1st slice

# Plot
plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140)
plt.title('Time Spent Planning Vacations vs. Retirement (Ages 26-41)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the plot
plt.show()
