from xarm.wrapper import XArmAPI
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

'''
    Gizzmos are external hardware elements that assist the robotic arm
    complete the PCR challenge! This list includes ECLAIR's specially
    curated gripper, a pressure sensor, and the Atalanta Module which
    uses OCR to adjust the volume on the pipettes to the desired amount
    for the reactant
'''

class CustomGripper:

    def __init__(self, arm: XArmAPI) -> None:
        self.arm = arm
        pass

    def close_gripper(self) -> str:
        return None, None

    def open_gripper(self) -> str:
        return None, None

    def fill_pipette(self) -> str:
        return None, None

    def empty_pipette(self) -> str:
        return None, None

    def remove_pipette_tip(self) -> str:
        return None, None

class PressureSensor:

    def __init__(self):
        pass

class AtalantaModule:

    def __init__(self):
        pass

    def adjust_volume(self, volume: float) -> str:

        # Communicate with arduino/RPI externally somehow
        # TCP Network socket maybe?
        return None, None

    def check_connection(self) -> bool:
        return True
    
'''
    Data model and loader for OCR on the pipette dial 
    - model located in 'mnist1-ce-sgd-3.pt'
'''

input_size = 784 # 28 * 28 input image size
hidden_sizes = [128, 64]
output_size = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        self.last_layer = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        logits = self.model(x)
        logits = self.last_layer(logits)
        return logits

    def predict(self, x):
        logits = self.model(x)
        logits = self.last_layer(logits)
        softmax = torch.nn.Softmax(dim=1)
        return softmax(logits)

# get input image
# replace image path with your image
image_path = 'printed_digit/train/3/0_0_7.jpeg'
# processing -> resize and turn B&W (white on black)

og_img = Image.open(image_path)
img = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize((28, 28)),
                            transforms.Grayscale(1),
                            transforms.Normalize((0.5,), (0.5,))])(og_img)

input = img.reshape(-1, 784) 

# load model
model_dir = 'models/transfer-ce-sgd-4.pt'
state = torch.load(model_dir)
model = Net()
try:
    state.eval()
except AttributeError as error:
    print(error)

model.load_state_dict(state['model_state_dict'])
model.eval()

# make prediction
with torch.no_grad():
    output = model.predict(input)

print(output)

print(torch.argmax(output))