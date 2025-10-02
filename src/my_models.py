from transformers import Dinov2Config, DPTForDepthEstimation, DPTConfig
import torch
from mmcv.cnn import build_upsample_layer

def dpt():

    # backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-large",
    #                                                out_features=["stage1", "stage2", "stage3", "stage4"],
    #                                                reshape_hidden_states=False
    #                                                )

    # config = DPTConfig(backbone_config=backbone_config)
    # model = DPTForDepthEstimation(config=config)
    
    # model = DPTForDepthEstimation.from_pretrained('/home/apolyubin/shared_data/SatelliteTo3D-Models/dpt/dfc2018')
    # model = model.train()
    
    model = DPTForDepthEstimation.from_pretrained('intel/dpt-large')
    model = model.train()
    return model

class CustomDPT():
    def __init__(self) -> None:
        super().__init__()

        # load backbone - dino model
        self.dpt_model = DPTForDepthEstimation.from_pretrained('Intel/dpt-large')


        # additional upscale layers
        self.additional_decoder_block = [
            build_upsample_layer(
                {'type': "deconv"},
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(inplace=True),
        ]
        self.additional_decoder_block = torch.nn.Sequential(*self.additional_decoder_block)

    def forward(self, img):
        """
        Forward pass of the DinoModel.

        Args:
            img (Tensor): The input image tensor.

        Returns:
            Tensor: The output depth tensor.
        """
        # backbone
        x = self.dpt_model(img)

        x = self.additional_decoder_block(x)

        return x
