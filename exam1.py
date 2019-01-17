import torch
import mlflow
import mlflow.pytorch
# X data
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# Y data with its expected value: labels
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
# Partial Model example modified from Sung Kim
# https://github.com/hunkim/PyTorchZeroToAll
class Model(torch.nn.Module):
    def __init__(self):
       super(Model, self).__init__()
       self.linear = torch.nn.Linear(1, 1)  # One in and one out
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
# our model
model = Model()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

mlflow.set_tracking_uri('http://192.168.0.220:5000')
mlflow.set_experiment('OCR')
with mlflow.start_run() as run:
    testtt = mlflow.get_tracking_uri()
    testttt = mlflow.get_artifact_uri()
    f = open('hello.txt', 'w')
    f.write('hello')
    f.close()
    # mlflow.log_artifact('hello.txt')

    mlflow.log_param("epochs", "10")
    mlflow.log_param("TrainData", 'COCO-train')
    mlflow.log_param("ValidData", 'COCO-valid')
    mlflow.log_param("TestData", 'COCO-test')
    mlflow.log_param("BatchSize", 'COCO-test')
    mlflow.log_param("Optimizer", 'Adadelta')
    mlflow.log_param("learning rate", '1e-1')

    # Training loop
    for epoch in range(10):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x_data)
        # Compute and print loss
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())
        #Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mlflow.log_metric("loss", loss.item())
        # mlflow.pytorch.log_model(model, 'model'+str(epoch))
    # After training
    for hv in [4.0, 5.0, 6.0]:
        hour_var = torch.Tensor([[hv]])
        y_pred = model(hour_var)
        print("predict (after training)",  hv, model(hour_var).data[0][0])
    # log the model
    # mlflow.log_param("epochs", 10)
    # mlflow.log_param("TrainData", 'COCO-train')
    # mlflow.log_param("ValidData", 'COCO-valid')
    # mlflow.log_param("TestData", 'COCO-test')
    # mlflow.pytorch.log_model(model, "models")