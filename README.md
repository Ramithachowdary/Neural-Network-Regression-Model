# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## Problem Statement and Dataset
The dataset contains customer information including Age and Spending Score. The goal is to predict the Spending Score of a customer based on their Age using a Neural Network Regression Model. Since the Spending Score is a continuous numerical value assigned based on customer purchasing behavior, this is treated as a regression problem. A feedforward neural network is used to learn the non-linear relationship between Age and Spending Score, trained using MSE loss and RMSprop optimizer, and evaluated on unseen test data to measure generalization.

## Neural Network Model
<img width="1134" height="647" alt="418446260-84093ee0-48a5-4bd2-b78d-5d8ee258d189" src="https://github.com/user-attachments/assets/f9a07d0f-c01e-4a3b-9ac3-bd8751e0f6cc" />

## DESIGN STEPS

### STEP 1: Loading the Dataset
The customer dataset is loaded using the Pandas library and the Age column is used as input while Spending Score is used as the target variable.

### STEP 2: Splitting the Dataset
The dataset is split into training and testing sets using an 67-33 ratio to evaluate model performance on unseen data.

### STEP 3: Data Scaling
MinMaxScaler is applied to normalize the input Age values between 0 and 1 to improve training stability.

### STEP 4: Building the Neural Network Model
A feedforward neural network is built using PyTorch with two hidden layers of size 8 and 10 neurons respectively, using ReLU activation functions and a single output neuron for regression.

### STEP 5: Training the Model
The model is trained for 2000 epochs using Mean Squared Error loss and RMSprop optimizer with a learning rate of 0.001.

### STEP 6: Plotting the Performance
Training loss is recorded at each epoch and plotted against iterations to visualize the learning behavior and convergence.

### STEP 7: Model Evaluation
The trained model is evaluated on the test set and test loss is printed. A new sample input is passed through the model to generate a predicted Spending Score.

## PROGRAM

### Name: Ramitha Chowdary S
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
print('Name: Ramitha Chowdary S')
print('Register Number: 212224240130')
loss_df = pd.DataFrame(ai_brain.history)
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.show()

# New Sample Prediction
print('Name: Ramitha Chowdary S')
print('Register Number: 212224240130')

X_new = torch.tensor([[30]], dtype=torch.float32)
X_new_scaled = torch.tensor(scaler.transform(X_new), dtype=torch.float32)

with torch.no_grad():
    prediction = ai_brain(X_new_scaled).item()

print(f"Input Age: 30")
print(f"Predicted Spending Score: {prediction:.2f}")
```

## OUTPUT

## Dataset Information

<img width="954" height="669" alt="image" src="https://github.com/user-attachments/assets/1cb1b3fe-aa46-4899-b50b-3572dadcd3e2" />

## OUTPUT

### Training Loss Vs Iteration Plot
<img width="489" height="570" alt="image" src="https://github.com/user-attachments/assets/7ae84479-d109-4114-9ca7-60eb458c01b6" />

### New Sample Data Prediction
<img width="232" height="65" alt="image" src="https://github.com/user-attachments/assets/e2f3509d-07aa-4060-b553-b06b21d0d8e7" />


## RESULT
The neural network regression model was successfully developed and trained on the customers dataset. The model learned to predict the Spending Score from the Age feature with decreasing training loss over 2000 epochs, confirming effective learning. The test loss validated the model's ability to generalize to unseen data and the sample prediction produced a reasonable Spending Score for the given input age.
