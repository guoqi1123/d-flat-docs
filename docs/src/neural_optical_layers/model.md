#


## MLP_Latent_Layer
[source](https://github.com/DeanHazineh/DFlat_Private/blob/master/DFlat_Private/src/neural_optical_layer/neural_optical_layers.py/#L11)
```python 
MLP_Latent_Layer(
   model_name
)
```


---
Neural-Optical Cell Model Layer; Initialized to call one of D-Flats pre-trained MLPs. This layer computes the
optical modulation (zero-order transmittance and phase) for cells, at user requested wavelengths, given a latent
vector input. For input of the normalized parameters, rather than the latent vector, use MLP_Layer instead. 

Once initialized with a MLP selection, this class may be recalled to evaluate different latent tensors.


**Attributes**

* MLP object/class initialized in the layer
* Input mlp shape of (1,D+1), where D is the shape degree and an extra column 
    specifies wavelength.  
* MLP trans and phase output batch size (==1 for polarization insensitive or ==2 for 
    polarization sensitive optics)



**Methods:**


### .initialize_input_tensor
[source](https://github.com/DeanHazineh/DFlat_Private/blob/master/DFlat_Private/src/neural_optical_layer/neural_optical_layers.py/#L90)
```python
.initialize_input_tensor(
   init_type, gridShape, dtype = tf.float64, init_args = []
)
```

---
Initialize a latent_tensor input. Valid initializations here are "uniform" and "random". To use an 
alternative, user-defined starting latent_tensor, one may be able to create their own 
using mlp_initialization_utilities.optical_response_to_param and a suitable param_to_latent call.


**Args**

* Selection of initialization types, either "uniform", "random"
* 2D cell grid shape given as a length three list, usually of the form [1, ms_samplesM["y"], ms_samples["x"]] or [1, 1, ms_samples["r"]]. 
* Data-type for the returned tensor. Defaults to tf.float64


**Returns**

* Latent_tensor of suitable form to pass to mlp_latent_layer call function.

