# Used to specify and instantiate the backbone that's going to be used for training
import timm
from huggingface_hub import login


def get_backbone(backbone_name: str, architecture: str, num_classes: int, pretrained: bool):
    """
        returns the shared backbone which the model is going to use according to the multi-domain multi-task training
        paradigm employed for this model training

        param backbone_name: denominates the backbone, used to determine which model is going to be loaded
        param architecture: string that tells timm which model to load
        param num_classes: the number of classes the final layer after the backbone should have (discarded)
        param pretrained: whether to use a pretrained version of the backbone or not
        """
    access_token = '<access_token>'
    login(access_token)
    match backbone_name:
        case 'resnet18':
            backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)
            num_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0)  # Remove the classifier
            return backbone, num_features
        case 'resnet50':
            backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)
            num_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0)  # Remove the classifier
            return backbone, num_features
        case 'densenet121':
            backbone = timm.create_model(architecture, pretrained=pretrained)
            num_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0)  # Remove the classifier
            return backbone, num_features
        case 'vgg16':
            backbone = timm.create_model(architecture, pretrained=pretrained)
            num_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0)  # Remove the classifier
            return backbone, num_features
        case 'vit_base_patch16_224':
            backbone = timm.create_model(architecture, pretrained=pretrained)
            num_features = backbone.num_features
            backbone.reset_classifier(0)  # Remove the classifier
            return backbone, num_features

    raise ValueError(f"backbone_name {backbone_name} did not match any of the expected architectures")
