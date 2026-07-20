import torch
from model import Net
import medmnist
from medmnist import INFO, Evaluator
from medmnist import OCTMNIST
import torch.utils.data as data
import torchvision.transforms as transforms

model_api = Net(in_channels=1, num_classes=4)
weights = torch.load("octmnist_cnn_weights.pth", map_location=torch.device('cpu'))
model_api.load_state_dict(weights)

BATCH_SIZE = 128
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])
test_dataset = OCTMNIST(split="test", download=True, transform=data_transform)

test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# evaluation
def test(split, loader):
  model_api.eval()
  y_true = torch.tensor([])
  y_score = torch.tensor([])

  with torch.no_grad():
    for inputs, targets in loader:
      outputs = model_api(inputs)
      outputs = outputs.softmax(dim=-1)
      
      targets = targets.squeeze().long()

      y_true = torch.cat((y_true, targets), 0)
      y_score = torch.cat((y_score, outputs), 0)

  y_true = y_true.numpy()
  y_score = y_score.detach().numpy()

  # y_true debe tener la forma (N,) y y_score (N, 4)
  print(f"Forma final -> y_true: {y_true.shape}, y_score: {y_score.shape}")

  evaluator = Evaluator("octmnist", split)
  metrics = evaluator.evaluate(y_score)

  print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))

        
print('==> Evaluating ...')
test('test', test_loader)