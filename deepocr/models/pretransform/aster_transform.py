import torch.nn as nn

from ..builder import PRETRANSFORMS
from .stn_head import ASTERSTNHead
from .tps import ASTERTPS


@PRETRANSFORMS.register_module()
class ASTERTransform(nn.Module):
    """Implementation of Transformation Module from `ASTER: An Attentional Scene Text Recognizer
    with Flexible Rectification`<https://ieeexplore.ieee.org/document/8395027>.
    """

    def __init__(
        self,
        in_channels,
        output_image_size,
        num_control_points=20,
        activation="none",
        margins=(0.05, 0.05),
    ):
        """
        Args:
            in_channels (int): Number of channel size of the input image.
            output_image_size (tupe or list): Output size of the image. Height first.
            num_control_points (int, optional): Total number of control points. Defaults to 20.
            activation (str, optional): Activation function if used. Defaults to "none".
            margins (tuple, optional): Margine for the transformed image. Defaults to (0.05, 0.05).
        """
        super(ASTERTransform, self).__init__()
        self.output_image_size = output_image_size
        self.stn_head = ASTERSTNHead(
            in_channels=in_channels,
            num_control_points=num_control_points,
            width=output_image_size[1],
        )
        self.tps = ASTERTPS(
            output_image_size=output_image_size,
            num_control_points=num_control_points,
            margins=margins,
        )

    def forward(self, x):
        ctrl_points = self.stn_head(x)
        x = self.tps(x, ctrl_points)

        return x
