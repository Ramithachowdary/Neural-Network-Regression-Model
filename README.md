# Developing a Neural Network Regression Model

## AIM

To develop a Neural Network–based regression model for predicting the spending score of customers using the given dataset.

## THEORY
```
Problem Statement Explanation:

An automobile company wants to understand customer behavior based on demographic features such as Age.
The Spending Score represents how actively a customer spends money. Predicting this score helps the company:
-Analyze customer purchasing behavior
-Design personalized offers
-Improve marketing strategies
Since the output (Spending Score) is a continuous numerical value, the problem is treated as a regression problem.
A Neural Network Regression Model is used to learn the relationship between age and spending score.
```
## Neural Network Model
<img width="1134" height="647" alt="418446260-84093ee0-48a5-4bd2-b78d-5d8ee258d189" src="https://github.com/user-attachments/assets/f9a07d0f-c01e-4a3b-9ac3-bd8751e0f6cc" />

## DESIGN STEPS

### STEP 1: Loading the Dataset
The customer dataset is loaded using the Pandas library.

### STEP 2: Splitting the Dataset
The dataset is split into training data and testing data to evaluate model performance.

### STEP 3: Data Scaling
MinMaxScaler is used to normalize the input values between 0 and 1.

### STEP 4: Building the Neural Network Model
A feedforward neural network is created using PyTorch with linear layers and ReLU activation.

### STEP 5: Training the Model
The model is trained using Mean Squared Error loss and RMSprop optimizer.

### STEP 6: Plotting the Performance
Training loss is plotted against epochs to visualize learning behavior.

### STEP 7: Model Evaluation
The trained model is evaluated using test data and test loss is calculated.

## PROGRAM

### Name: Ramitha chowdary S
### Register Number: 212224240130

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load Dataset
dataset1 = pd.read_csv('customers.csv')

X = dataset1[['Age']].values
y = dataset1[['Spending_Score']].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize Model, Loss and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)

# Training Function
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()
        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

# Train the Model
train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

# Test Evaluation
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")

# Plot Loss
loss_df = pd.DataFrame(ai_brain.history)
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.show()

# New Sample Prediction
X_new = torch.tensor([[9]], dtype=torch.float32)
X_new_scaled = torch.tensor(scaler.transform(X_new), dtype=torch.float32)

prediction = ai_brain(X_new_scaled).item()
print(f"Predicted Spending Score: {prediction}")
```
## Dataset Information

<img width="954" height="669" alt="image" src="https://github.com/user-attachments/assets/1cb1b3fe-aa46-4899-b50b-3572dadcd3e2" />

## OUTPUT

### Training Loss Vs Iteration Plot
<img width="601" height="595" alt="image" src="https://github.com/user-attachments/assets/0e379cd1-b1bf-47f4-9afe-14095a54caba" />

### New Sample Data Prediction
<img width="613" height="243" alt="image" src="https://github.com/user-attachments/assets/49251c6c-7e0c-4070-8c12-a58ff673ab3e" />

<img width="715" height="97" alt="image" src="https://github.com/user-attachments/assets/907fac51-67e8-45f2-a8ed-c99c174fd495" />

## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
