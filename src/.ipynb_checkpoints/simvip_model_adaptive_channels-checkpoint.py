import torch
from torch import nn

from modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock, ConvSC_ReLUResNet)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class SimVP_Model_no_skip_sst(nn.Module):
    r"""SimVP Model
    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.
    """

    def __init__(self, in_shape, in_channels=2, out_channels=1, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        """
        Initialize the SimVP model without skip connections.
        
        Args:
            in_shape (tuple): Input shape as (T, C, H, W), where T is the sequence length,
                             C is the number of channels, H and W are height and width.
            hid_S (int): Number of hidden spatial channels.
            hid_T (int): Number of hidden temporal channels.
            N_S (int): Number of spatial encoding layers.
            N_T (int): Number of temporal encoding layers.
            model_type (str): Type of mid-level model ('gSTA', 'Incepu', etc.).
            mlp_ratio (float): Ratio for MLP layers in the mid-level model.
            drop (float): Dropout rate.
            drop_path (float): Stochastic depth drop path rate.
            spatio_kernel_enc (int): Kernel size for spatial encoder.
            spatio_kernel_dec (int): Kernel size for spatial decoder.
            act_inplace (bool): Whether to perform in-place activation.
        """
        super(SimVP_Model_no_skip_sst, self).__init__()
        # Define the number of input and output channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Extract dimensions from input shape
        T, C, H, W = in_shape  # T is the temporal sequence length
        # Downsample height and width based on spatial layers
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # Reduce resolution progressively
        # Force activation function not to be in-place
        act_inplace = False
        # Define two spatial encoders without skip connections
        self.encoders = nn.ModuleList([
                            Encoder_no_skip(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
                            for _ in range(in_channels)
                            ])
        # Define decoder that reconstructs the input
        self.dec = Decoder_no_skip(in_channels * hid_S, out_channels, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        # Convert model type to lowercase if provided
        model_type = 'gsta' if model_type is None else model_type.lower()
        # Define the temporal processing module based on model type
        if model_type == 'incepu':
            self.hid = MidIncepNet(T * hid_S * 2, hid_T, N_T)  # Inception-based temporal model
        else:
            self.hid = MidMetaNet(T * hid_S * 2, hid_T, N_T,
                                  input_resolution=(H, W), model_type=model_type,
                                  mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)  # MetaNet-based temporal model

    def forward(self, x_raw, **kwargs):
        """
        Forward pass of the SimVP model.
        
        Args:
            x_raw (torch.Tensor): Input tensor of shape (B, T, C, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, T, 1, H, W)
        """
        # Extract batch size and input dimensions
        B, T, C, H, W = x_raw.shape
        # Flatten temporal dimension into batch dimension (treat each frame separately initially)
        x = x_raw.view(B * T, C, H, W)
        # Encode the different channels using different encoders
        embeds = [
            encoder(x[:, i, :, :].reshape(B * T, 1, H, W))
            for i, encoder in enumerate(self.encoders)
            ]
        embed = torch.cat(embeds, dim=1)
        # Get new shape after encoding
        _, C_, H_, W_ = embed.shape
        # Reshape back into a temporal batch structure
        z = embed.view(B, T, C_, H_, W_)
        # Pass through the temporal model
        hid = self.hid(z)
        # Reshape back into (B*T, C_, H_, W_) for decoding
        hid = hid.reshape(B * T, C_, H_, W_)
        # Decode to reconstruct the output
        Y = self.dec(hid)
        # Reshape output back to (B, T, 1, H, W)
        Y = Y.reshape(B, T, self.out_channels, H, W)
        
        return Y
        

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sampling_generator(N, reverse=False):
    """
    Generates a list of alternating True and False values, used for determining
    downsampling or upsampling in the encoder/decoder.
    
    Args:
        N (int): Number of elements in the list.
        reverse (bool): If True, reverses the sequence order.
    
    Returns:
        list: A list of alternating boolean values.
    """
    samplings = [False, True] * (N // 2)
    return list(reversed(samplings[:N])) if reverse else samplings[:N]


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Encoder_no_skip(nn.Module):
    """3D Encoder for SimVP without skip connections."""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        """
        Initializes the encoder module.
        
        Args:
            C_in (int): Number of input channels.
            C_hid (int): Number of hidden channels.
            N_S (int): Number of spatial layers.
            spatio_kernel (int): Kernel size for convolution layers.
            act_inplace (bool): Whether to use in-place activation.
        """
        samplings = sampling_generator(N_S)
        super(Encoder_no_skip, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B*4, 3, 128, 128)
        
        Returns:
            torch.Tensor: Encoded feature map.
        """
        latent = self.enc[0](x)
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Decoder_no_skip(nn.Module):
    """3D Decoder for SimVP without skip connections."""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        """
        Initializes the decoder module.
        
        Args:
            C_hid (int): Number of hidden channels.
            C_out (int): Number of output channels.
            N_S (int): Number of spatial layers.
            spatio_kernel (int): Kernel size for convolution layers.
            act_inplace (bool): Whether to use in-place activation.
        """
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder_no_skip, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid):
        """
        Forward pass through the decoder.
        
        Args:
            hid (torch.Tensor): Encoded feature map.
        
        Returns:
            torch.Tensor: Reconstructed output tensor.
        """
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid)
        Y = self.readout(Y)
        return Y


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class MidIncepNet(nn.Module):
    """The hidden translator of IncepNet for SimVPv1."""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11], groups=8, **kwargs):
        """
        Initializes the MidIncepNet module.
        
        Args:
            channel_in (int): Number of input channels.
            channel_hid (int): Number of hidden channels.
            N2 (int): Number of layers in the network.
            incep_ker (list): Kernel sizes for the inception layers.
            groups (int): Number of groups for grouped convolutions.
        """
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        
        # Define encoder layers
        enc_layers = [gInception_ST(channel_in, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for _ in range(1, N2-1):
            enc_layers.append(gInception_ST(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(gInception_ST(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        
        # Define decoder layers
        dec_layers = [gInception_ST(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for _ in range(1, N2-1):
            dec_layers.append(gInception_ST(2*channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(gInception_ST(2*channel_hid, channel_hid//2, channel_in, incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W).
        
        Returns:
            torch.Tensor: Processed output tensor of the same shape.
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # Encoder pass
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        
        # Decoder pass
        z = self.dec[0](z)
        for i in range(1, self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))
        
        y = z.reshape(B, T, C, H, W)
        return y


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type in ['mlp', 'mlpmixer']:
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type in ['moga', 'moganet']:
            self.block = MogaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(
                in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path, block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y