import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(r'student_hours_marks_1000.csv')

marks = data['marks']
hours_studied = data['hours_studied']
n = len(marks)
learning_rate = 0.01

iterations = 1000

w=b=0

for i in range(iterations):
    y_bar = w*hours_studied + b

    loss = (1/n)*np.sum((y_bar - marks)**2)

    dw = (2/n)*np.sum(hours_studied*(y_bar - marks))

    db = (2/n) * np.sum(y_bar - marks)

    w = w - learning_rate*dw

    b = b - learning_rate*db

y_bar = w * hours_studied + b

test_input = int(input("Enter number of hours studied"))

print(f"The predicted marks are {w * test_input + b} out of {100}")

sorted_indices = np.argsort(hours_studied)

x_sorted = hours_studied.iloc[sorted_indices]
y_sorted = y_bar.iloc[sorted_indices]

plt.scatter(hours_studied, marks)
plt.plot(x_sorted, y_sorted)

plt.xlabel('No. of Hours studied')
plt.ylabel('Marks obtained')
plt.title('Linear Regression')

plt.show()

