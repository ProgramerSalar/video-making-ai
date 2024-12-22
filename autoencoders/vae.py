from typing import Optional, Union

import torch
import inspect
import math 
import torch.nn as nn 
from diffusers import ConfigMixin, ModelMixin
from autoencoders.causal_nd_factory import make_conv_nd
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput

# ---------------------------------------------------------------------------------------------------------------------------------

## class AutoencoderKLWrapper(ModelMixin, ConfigMixin)

# fn init (encoder: nn.Module, decoder, latent_channel=4, dims=2, sample_size=512, use_quant_conv)
#     |
# init params to Encoder 
#     |
# pass init params to Decoder 
#     |
# make a condition of use_quantize_conv 
#     |
# pass the data in make_conv_nd and object is quant_conv also similar to pos_quant_conv 
#     |
# conditin else pass the identity function 
#     |
# get some object [use_z_tilling=false, use_hw_tilling=false, dims, z_sample_size=1]
#     |
# create decoder parameters obj is decoder_parameters 
#     |
# use this fn set_tilling_params

# ___________________________________________________________________________________________________________________________________________

# fn set_tilling_params(sample_size: int = 512, overlap_factor = 0.25)
#     |
# create a object tile_sample_min_size and num_blocks = len(self.encoder.down_blocks)
#     |
# create obj tile_latent_min_size and tile_overlap_factor 

# ____________________________________________________________________________________________________________________________________________

# fn enable_z_tilling (z_sample_size: int = 8)

# fn disable_z_tilling 

# fn enable_hw_tilling 

# fn disable_hw_tilling 

# _____________________________________________________________________________________________________________________________________________

# fn blend_v (a: torch.Tensor, b, blend_extent: int) -> torch.tensor
#     |
# min of a, b and blend_extent shape 
#     |
# iteration of blend_extent 
#     |
# from b change the height dim =  data [a] change in height dim  -blend_extent + y * (1 - y / blend_extent) + change of b in height dim * (y / blend_extent)
#     |
# return b 


# fn blend_h 

# fn blend_z 


# ________________________________________________________________________________________________________________________________________________

# fn _hw_tile_encode(x: FloatTensor, target_shape)
#     |
# create overlap_size (tile_latent_min_size * (1 - 0.25))
#     |
# create blend_extent (512 * 0.25)
#     |
# create row_limit (512 - blend_extent)
#     |
# first iteration of range zero to x-shape[3] and step is overlap_size 
#     |
# second iteration of range zero to x-shape[4] and step is overlap_size 
#     |
# in data x change in horizontal and veritical added the size of 512 
#     |
# encode and quantize convolution them
#     |
# append the tile in row list 
#     |
# append the row in rows list and dim is 3
#     |
# [Again Iteration ]
#     |
# extract i and row in rows list using the iteration method 
#     |
# extract j and tile in row list using the iteration method 
#     |
# if i is greater then 0 then blend vertical size  pass the object is tensor a = rows[i - 1][j], and tensor b is tile 
#     |
# if j is greater than 0 then blend horizontal size pass the object is tensor a = row[j - 1] and tensor b = tile 
#     |
# append in the result_row the tile height and width for maximum size is row_limit,   
#     |
# result_row append in result_rows in dimension of 3 
#     |
# result moment have dimension = 3 
#     |
# return moment


# _________________________________________________________________________________________________________________________________________________

# fn _hw_tile_decode(z, target_shape)
#     |
# same for above onely changes things are write the down line 
#     |
# tile_target_shape is (*target_shape[3:], self.tile_sample_min_size, self.tile_latent_min_size)
#     |
# before decode use the post quantize convolution pass the tile 
#     |
# in decode pass the tile and target_shape = tile_target_shape

# ____________________________________________________________________________________________________________________


# fn _encode(x: floatTensor) -> AutoencoderKLOutput
#     |
# create a object h have encoder pass the x 
#     |
# create a object moments have quantize convolution pass h 
#     |
# return moments

# __________________________________________________________________________________________________

# fn encode (z, return_dict=True) -> Union[DecoderOutput, torch.FloatTensor]
#     |
# [condition if use_z_tilling is True and shape of z[2] is greater than z_sample_size is greater than 1 ]
#     |
# create num_splits = shape of z[2] divisible of z_sample_size 
#     |
# create sizes = list of z_sample_size multiply of num_splits
#     |
# object sizes = sizes + list of shape of z[2] minus sum of sizes [if shape of z[2] - sum of sizes is greater than 0] else sizes 
#     |
# create tiles = split of z into sizes of dim=2
#     |
# condition if use_hw_tilling is True then return the _hw_tile_encode(z_tile, return_dict) 
# else _encode(z_tile) this condition run in the loop of tiles this loop store in list of moments_tiles
#     |
# concatenate of moments_tiles in dimension of 2 object is moments 
#     |
# [condition else ]
#     |
# if use_hw_tilling is True then return _hw_tile_encode (z, return_dict) else _encode(z) store this condition in tuple of moments 
#     |
# posterior = DiagonalGaussianDistribution(moments)
#     |
# condition if return_dict is not found return the tuple of posterior 
#     |
# return AutoencoderKLOutput(latent_dist= posterior)

# _________________________________________________________________________________________________________________________________________________

# fn _decode (z, target_shape = None, timesteps: Optional[torch.Tensor] = None) -> Union[DecoderOutput, torch.floatTensor]
#     |
# object z have post_quantize_convolution pass the z 
#     |
# if "timesteps" have in decoder_parameters return the decoder(z, target_shape, timesteps)
#     |
# else decoder(z, target_shape)
#     |
# return dec 

# ___________________________________________________________________________________________________________________________________________


# fn decoder (same as _decoder, return_dict: bool = True) 
#     |
# assert target_shape is not None return "target shape must be provided for decoding"
#     |
# [condition if use_z_tilling is True and shape of z[2] is greater than z_sample_size is greater than 1 ]
#     |
# patch_size_t in encoder multiple of 2 and square length of encoder down_blocks - 1 - root of endoer 
# patch_size make sure all are integer and store in object reduction_factor 
#     |
# create split_size = z_sample_size divisible of reduction_Factor 
#     |
# create num_splits = shape of z[2] divisible of split size  
#     |
# create target_shape_split is list of target_shape 
#     |
# target_shape_split of index 2 = target_shape of index 2 divisible of num_splits 
#     |
# iteration of tensor_split(z, num_splits, dim=2) in z_tile if use_hw_tilling is true 
# return _hw_tiled_decode(z_tile, target_shape_split) else _decode(z_tile, target_shape_split) 
# this things are store in list of decoded_tiles 
#     |
# concatenate of decoded_tiles dim=2  object decoded 
#     |
# [condition else ]
#     |
# if use_hw_tiling is true return the _hw_tiled_decoded(z, target_shape) else _decode(z, target_shape, timesteps) this things are store in tuple of decoded 
#     |
# condition if return_dict is not found return the tuple of decoded 
#     |
# return DecoderOutput(decoded)

# _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# fn forward(sample:float, saple_posterior=False, return_dict=True, generator: Optional[torch.Genertor] = None) -> Union[DecoderOutput, floattensor]
#     |
# x = sample 
#     |
# create a object posterior = pass the x in encode then use the latent_dist 
#     |
# if sample_posterior is true return the pass the generator in posterior.sample  object is z 
#     |
# else mode of posterior 
#     |
# object dec = decode(z, sample.shape).sample 
#     |
# condition if return_dict is not found return the tuple of dec 
#     |
# return DecoderOutput(dec)

# ---------------------------------------------------------------------------------------------------------------------------------


class AutoencoderKLWrapper(ModelMixin, ConfigMixin):

    def __init__(self,
                encoder: nn.Module,
                decoder: nn.Module,
                latent_channel = 4,
                dims = 2,
                sample_size = 512,
                use_quant_conv:bool=False
                ):


        # init params to Encode 
        self.encoder= encoder
        self.use_quant_conv = use_quant_conv


        # init params to Decoder 
        self.decoder = decoder

        # make a condition of use_quantize_conv 
        quant_dims = 3 if dims == 3 else 2


        if use_quant_conv:
            
            self.quant_conv = make_conv_nd(dims=quant_dims,
                                            in_channels= 2 * latent_channel,
                                            out_channels= 2 * latent_channel,
                                            kernel_size=1)


            self.post_quant_conv = make_conv_nd(dims=quant_dims,
                                                in_channels= 2 * latent_channel,
                                                out_channels= 2 * latent_channel,
                                                kernel_size= 1)


        else:
            self.quant_conv = torch.nn.Identity()
            self.post_quant_conv = torch.nn.Identity()



        self.use_z_tilling = False
        self.use_hw_tilling = False
        self.dims = dims
        self.z_sample_size = 1 


        self.decoder_params = inspect.signature(self.decoder.forward).parameters


        # Daily relevent if vae tilling is enabled 
        self.vae_tilling_enabled(sample_size=sample_size,
                                overlap_factor=0.25)



    def vae_tilling_enabled(self,
                            sample_size: int = 512,
                            overlap_factor = 0.25):
        
        self.tile_sample_min_size = sample_size
        self.num_blocks = len(self.encoder.down_blocks)

        self.tile_latent_min_size = int(sample_size / (2 * (self.num_blocks - 1)))
        self.tile_overlap_factor = overlap_factor



    def enable_z_tilling(self,
                z_sample_size: int = 8):

        self.z_sample_size = z_sample_size > 1 
        self.z_sample_size = z_sample_size

        assert (
            z_sample_size % 8 == 0 or z_sample_size == 1
        ), f"z_sample_size must be a multiple of 8 or 1."



    def disable_z_tilling(self):

        self.use_z_tilling = False


    def enable_hw_tilling(self):
        
        self.use_hw_tilling = False


    def disable_hw_tilling(self):
        self.use_hw_tilling = False



    def blend_v(self,
                a: torch.Tensor,
                b: torch.Tensor,
                blend_extent: int) -> torch.Tensor:


        blend_extent = min(a.shape[3], b.shape[3], blend_extent)

        for i in blend_extent:

            b[:, :, :, i, :] = a[:, :, :, -blend_extent + i, :] * (
                1 - i / blend_extent
            ) + b[:, :, :, i, :] * (i / blend_extent)


        return b 


    def blend_h(self,
                a: torch.Tensor,
                b: torch.Tensor,
                blend_extent: int) -> torch.Tensor:


        blend_extent = min(a.shape[4], b.shape[4], blend_extent)

        for i in blend_extent:

            b[:, :, :, :, i] = a[:, :, :, :, -blend_extent + i] * (
                1 - i / blend_extent
            ) + b[:, :, :, :, i] * (i / blend_extent)


        return b 



    def blend_z(self,
                a: torch.Tensor,
                b: torch.Tensor,
                blend_extent: int) -> torch.Tensor:


        blend_extent = min(a.shape[2], b.shape[2], blend_extent)

        for i in blend_extent:
            b[:, :, i, :, :] = a[:, :, -blend_extent + i, :, :] * (
                1 - i / blend_extent
            ) + b[:, :, i, :, :] * (i / blend_extent)


        return b 



    def _hw_tile_encode(self, x: torch.FloatTensor, target_shape):

        overlap_size = min(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = min(self.tile_sample_min_size - self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[:, :, :, i: i + self.tile_sample_min_size, j: j + self.tile_sample_min_size]

                tile = self.encoder(tile)
                tile = self.quant_conv(tile)

                row.append(tile)

            rows.append(torch.cat(row, dim=3))

        
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):

                if i > 0:
                    tile = self.blend_v(a = rows[i - 1][j],
                                        b = tile,
                                        blend_extent=blend_extent)

                if j > 0:
                    tile = self.blend_h(a = row[j - 1],
                                        b = tile,
                                        blend_extent=blend_extent)



                result_row.append(tile[:, :, :, :row_limit, :row_limit])

            result_rows.append(torch.cat(result_row, dim=3))

        moment = torch.cat(result_rows, dim=4)

        return moment



    def _hw_tile_decode(self,
                        z: torch.FloatTensor,
                        target_shape):



        overlap_size = min(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = min(self.tile_sample_min_size - self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent
        tile_target_shape = (
            *target_shape[3:],
            self.tile_sample_min_size,
            self.tile_latent_min_size
        )


        rows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[:, :, :, i: i + self.tile_sample_min_size, j: j + self.tile_sample_min_size]

                tile = self.post_quant_conv(tile)
                decode = self.decoder(tile, target_shape=tile_target_shape)

                row.append(decode)

            rows.append(torch.cat(row, dim=3))

        
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):

                if i > 0:
                    tile = self.blend_v(a = rows[i - 1][j],
                                        b = tile,
                                        blend_extent=blend_extent)

                if j > 0:
                    tile = self.blend_h(a = row[j - 1],
                                        b = tile,
                                        blend_extent=blend_extent)



                result_row.append(tile[:, :, :, :row_limit, :row_limit])

            result_rows.append(torch.cat(result_row, dim=3))

        moment = torch.cat(result_rows, dim=4)

        return moment

            
    def _encode(self,
                x: torch.FloatTensor) -> AutoencoderKLOutput:


        h = self.encoder(x)
        moments = self.quant_conv(h)

        return moments


    def encode(self,
                z: torch.FloatTensor,
                return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:


        if self.use_z_tilling and z.shape[2] > self.z_sample_size > 1:
            
            num_splits = z.shape[2] // self.z_sample_size
            sizes = [
                self.z_sample_size * num_splits
            ]

            sizes = sizes + [
                z.shape[2] - sum(sizes) 
                if z.shape[2] - sum(sizes) > 0 
                else sizes
            ]

            tiles = torch.split(tensor=z,
                                dim=2)


            moments_tiles = [
            (
            
            self._hw_tile_encode(x= z_tile,
                                target_shape= return_dict)
            
            if self.use_hw_tilling
            else self._encode(x=z_tile)

            )
            for z_tile in tiles 

            ]


            moments = torch.cat(tensors=moments_tiles,
                                dim=2)
                  

        else: 

            moments = (

                
                self._hw_tile_encode(x = z,
                                    target_shape= return_dict)
                if self.use_hw_tilling
                else self._encode(x = z)

                )


        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior, )


        return AutoencoderKLOutput(latent_dist=posterior)



    def _decode(self,
                z: torch.FloatTensor,
                target_shape: float = None,
                timesteps: Optional[torch.Tensor] = None) -> Union[DecoderOutput, torch.FloatTensor]:


        

        z = self.post_quant_conv(z)

        if "timesteps" in self.decoder_params:
            return self.decoder(z, target_shape, timesteps)

        else:
            return self.decoder(z, target_shape)

        return dec 



    def decoder(self,
                z: torch.FloatTensor,
                return_dict: bool = True,
                target_shape: float = None,
                timesteps: Optional[torch.Tensor] = None) -> Union[DecoderOutput, torch.FloatTensor]:



        assert (target_shape is not None), "target_shape must be provided for decoding"

        if self.use_z_tilling and z.shape[2] > self.z_sample_size > 1:

            reduction_factor = int(self.encoder.patch_size_t * 2 ** (
                len(self.encoder.down_blocks) - 1 - torch.sqrt(self.encoder.patch_size)
            ))

            split_size = self.z_sample_size // reduction_factor
            num_splits = z.shape[2] // split_size 
            target_shape_split = [target_shape]
            target_shape_split[2] = target_shape[2] // num_splits


            decoded_tiles = [

                    (
                        
                        self._hw_tile_decode(z=z_tile, target_shape=target_shape_split)
                        if self.use_hw_tilling

                        else self._decode(z=z_tile, target_shape=target_shape_split)

                    )
                    for z_tile in torch.tensor_split(z, num_splits, dim=2)
                
            ]

            decoded = torch.cat(decoded_tiles, dim=2)       

        else:


            decoded = (
                
                
                self._hw_tile_decode(z=z, target_shape=target_shape)
                if self.use_hw_tilling

                else self._decode(z = z,
                            target_shape=target_shape,
                            timesteps=timesteps)

            )


        if not return_dict:
            return (decoded, )


        return DecoderOutput(decoded)



    def forward(self,
                sample: float,
                sample_posterior=False,
                return_dict = True,
                generator: Optional[torch.Generator] = None) -> Union[DecoderOutput, torch.FloatTensor]:



        x = sample

        posterior = self.encode(x).latent_dist
         
        if sample_posterior: 
            
            z = posterior.sample(generator)

        else:
            z = self.decoder(z, sample.shape).sample


        dec = self.decoder(z, sample.shape).sample


        if not return_dict:
            return (dec, )


        return DecoderOutput(dec)






        
        

            
            



            
        






        








            









    




