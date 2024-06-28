from torch import nn
from torchvision.models import efficientnet_b0  # You can change this to any EfficientNet variant

class EfficientNetModel(nn.Module):
    def __init__(self, dataset_class, output_size, dropout_rate: float = 0.2):
        super().__init__()
        self.model = efficientnet_b0(pretrained=True)
        
        # Adjust the first layer to handle single-channel input
        if dataset_class.channels == 1:  # Assuming dataset_class.channels provides the number of input channels
            self.model = self._modify_first_layer(self.model)

        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, output_size)
        )

    def _modify_first_layer(self, model):
        # Replace the first layer to handle single-channel input
        new_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.features[0][0] = new_conv
        return model

    def forward(self, x):
        return self.model(x)
