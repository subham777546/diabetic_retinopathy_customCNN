import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import io


class CombinedNet(nn.Module):
    """
    A combined model using pre-trained ResNet-50 and DenseNet-121 backbones,
    followed by a custom linear layer for classification into 5 DR grades (0-4).
    This class definition must exactly match the one used during training.
    """
    def __init__(self):
        super(CombinedNet, self).__init__()
        
        
        self.resnet = models.resnet50(pretrained=True)
        self.densenet = models.densenet121(pretrained=True)
        
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.densenet.parameters():
            param.requires_grad = False
            
       
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, 512)
        
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 256), # 512 (ResNet) + 512 (DenseNet) = 1024
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 5) # 5 classes (DR grades 0-4)
        )
        
    def forward(self, x):
        x1 = self.resnet(x)
        x2 = self.densenet(x)
        x = torch.cat((x1, x2), dim=1) 
        x = self.fc(x)
        return x


INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = CombinedNet()
MODEL_PATH = "best_combined_model.pth" # Ensure this file exists in the backend directory

try:
  
    MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval() # Set model to evaluation mode (crucial for inference)
    print(f"Model loaded successfully from {MODEL_PATH} on {DEVICE}.")
except Exception as e:
    MODEL = None
    print(f"ERROR LOADING MODEL: {e}")
    print("ACTION REQUIRED: Ensure 'best_combined_model.pth' is in the backend directory and the file is not corrupt.")
    


def predict_image(image_bytes: bytes) -> dict:
    """
    Processes an image (as bytes), runs inference using the loaded PyTorch model, 
    and returns the predicted class and confidence.
    """
    if MODEL is None:
        return {"prediction_class": -1, "confidence": "0.00", "error": "Model not available. Check server logs for loading error."}

    try:

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        

        input_tensor = INFERENCE_TRANSFORM(image)
        

        input_batch = input_tensor.unsqueeze(0) 
        input_batch = input_batch.to(DEVICE)
        

        with torch.no_grad():
            output = MODEL(input_batch)
        

        probabilities = nn.functional.softmax(output[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)
        

        return {
            "prediction_class": predicted_class.item(), 
            "confidence": f"{confidence.item() * 100:.2f}"
        }
        
    except Exception as e:
        print(f"Inference processing failed: {e}")
        return {"prediction_class": -1, "confidence": "0.00", "error": f"Inference error: {e}"}
