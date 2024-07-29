import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.random.randn(100, 2)
data = np.vstack([data, np.random.uniform(low=-6, high=6, size=(20, 2))])

# Fit the model
model = IsolationForest(contamination=0.1)
model.fit(data)
predictions = model.predict(data)

# Plot results
plt.scatter(data[:, 0], data[:, 1], c=predictions, cmap='coolwarm')
plt.title('Anomaly Detection with Isolation Forest')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
