# Functions for receptive field calculation of a keras model
# Code by Christian Schiffer, FZJ
# Email: c.schiffer@fz-juelich.de

def get_layer(model, layer):
    """
    Helper function to get a layer from a model. It supports getting the layer
    by index, by name or by reference.

    Args:
        model (keras.models.Model): Model to get the layer from.
        layer (str or int or keras.layers.Layer): Layer to receive. Can be a name, an index or the layer itself (for convenience).

    Returns:
        Found keras.layers.Layer. Raises an error if layer could not be found.
    """
    if isinstance(layer, int):
        return model.get_layer(index=layer)
    elif isinstance(layer, str):  
        return model.get_layer(name=layer)
    elif layer in model.layers:
        return layer
    else:
        raise RuntimeError("Could not find layer given {}".format(layer))

        
def get_layer_index(model, layer):
    """
    Get the index of a layer inside a model.

    Args:
        model (keras.models.Model): Model containing the layer.
        layer (str or int or keras.layers.Layer): Layer to get index for. Can be a name, an index or the layer itself (for convenience).

    Returns:
        Index of the layer inside the model.
    """
    return model.layers.index(get_layer(model, layer))


def is_list_like(obj):
    """
    Checks if obj is either a list or a tuple.
    """
    return isinstance(obj, (list, tuple))


def make_list_like(obj):
    """
    Makes a list out of an object, if it is not yet a list.    
    """
    return obj if is_list_like(obj) else [obj]

# noinspection PyUnresolvedReferences
def analyze_receptive_field(network, start_layer=None, stop_layer=None, scale_factor=1.0, receptive_field=0.0, output_constraint=(1, 0), first_call=True):
    """
    Recursively calculates the receptive field of a given network. Given the last layer of a neural network,
    it traverses backwards to the input layer and calculates the receptive field, a scale factor between input and
    output and an output constraint.

    Remark:
        Only square filters are supported. Also, for deconvolution only the case stride == size is supported.

    Args:
        network (keras.models.Model): Neural network model to calculate receptive field of.
        start_layer (keras.layers.Layer or list of keras.layers.Layer or None): Layer to start the calculation at. If this is None, start at the output layer(s).
        stop_layer (keras.layers.Layer or list of keras.layers.Layer or None): Layer to stop calculation at. If this is None, stop at the input layer(s).
        scale_factor (float): Current scale_factor. Will be modified when recursively iterating through the network.
        receptive_field (float): Current receptive field size. Will be modified when recursively iterating through the network.
        output_constraint (tuple): Current output constraint. Will be modified when recursively iterating through the network.
        first_call (bool): Indicates if this is the first call to the function (non-recursive). Don't set this from outside of this function itself!

    Returns:
        Dictionary with the keys...

        "scale_factor": Factor between input and output size. Useful for networks where the contracting branch is deeper than
                        the expanding path.
        "receptive_field": Number of pixels on each side which are taken into consideration calculating a output pixel value.
        "output_constraint": Tuple (c, f) with c and f such that output_size * f + c = input_size and input_size is a whole number.
                             The output constraint must hold for an input size to be valid.
    """
    import numpy as np
    from keras.layers import InputLayer

    if start_layer is None:
        start_layer = network.output_layers
    start_layer = make_list_like(start_layer)
    start_layer = [get_layer(model=network, layer=layer) for layer in start_layer]

    if stop_layer is None:
        stop_layer = network.input_layers
    stop_layer = make_list_like(stop_layer)
    stop_layer = [get_layer(model=network, layer=layer) for layer in stop_layer]

    if first_call:
        # Try to use cached receptive fields inside the model.
        cached_receptive_field = _get_receptive_field_from_cache(model=network, start_layer=start_layer, stop_layer=stop_layer)
        if cached_receptive_field:
            return cached_receptive_field

    current_layer = start_layer
    while True:
        if isinstance(current_layer, list) and len(current_layer) == 1:
            current_layer = current_layer[0]
        if isinstance(current_layer, list):
            # For layers with multiple input layers, traverse all possible backwards paths
            receptive_field_per_path = []
            for input_layer in current_layer:
                receptive_field_per_path.append(analyze_receptive_field(network=network,
                                                                        start_layer=input_layer,
                                                                        stop_layer=stop_layer,
                                                                        scale_factor=scale_factor,
                                                                        receptive_field=receptive_field,
                                                                        output_constraint=output_constraint,
                                                                        first_call=False))
            # Determine the largest receptive field of all paths backward through the network
            # noinspection PyTypeChecker
            final_receptive_field = receptive_field_per_path[np.argmax([r["receptive_field"] for r in receptive_field_per_path])]

            if first_call:
                # Add receptive field to cache (only for the top recursion call)
                _add_receptive_field_to_cache(model=network, receptive_field=final_receptive_field, start_layer=start_layer, stop_layer=stop_layer)

            return final_receptive_field
        elif not (isinstance(current_layer, InputLayer) or current_layer in stop_layer):
            # For layers with just one input_layer, calculate the receptive field
            size = 1
            stride = 1
            dilation = 1
            if is_conv2d(current_layer) or is_maxpool2d(current_layer):
                # Downscaling by convolution or pooling
                if hasattr(current_layer, "kernel_size"):
                    size = current_layer.kernel_size[0]
                elif hasattr(current_layer, "pool_size"):
                    size = current_layer.pool_size[0]
                if hasattr(current_layer, "strides"):
                    stride = current_layer.strides[0]
                if hasattr(current_layer, "dilation_rate"):
                    dilation = current_layer.dilation_rate[0]

                if size < stride:
                    raise RuntimeError("Error while calculating receptive field: "
                                       "Size must be equal or greater than stride (layer: {})".format(current_layer.name))

                scale_factor *= stride
                receptive_field *= stride
                receptive_field += (size - stride) * dilation
            elif is_deconv2d(current_layer):
                # Upscaling by deconvolution
                size = current_layer.kernel_size[0]
                stride = current_layer.strides[0]

                if size != stride:
                    # We don"t know how to threat this case, so we raise an error...
                    raise RuntimeError("Error while calculating receptive field: "
                                       "For deconvolution, size == stride must hold (layer: {})".format(current_layer.name))

                scale_factor /= stride
                receptive_field /= stride
                output_constraint = (scale_factor, receptive_field)
            elif is_upsampling2d(current_layer):
                # Upsampling by nearest neighbor interpolation.
                # Each pixel gets replicated N times, where N
                # is the size of the upsampling layer.
                upsampling_factor = current_layer.size[0]

                # This is equivalent to a deconvolution with size
                # and stride determined by the upsampling_factor
                # and weights fixed to 1, so each pixel gets simply
                # replicated.
                scale_factor /= upsampling_factor
                receptive_field /= upsampling_factor
                output_constraint = (scale_factor, receptive_field)

            new_layer = []
            # noinspection PyProtectedMember
            for inbound_node in current_layer._inbound_nodes:
                new_layer.extend(inbound_node.inbound_layers)

            current_layer = new_layer
        else:
            receptive_field = {"scale_factor": scale_factor,
                               "receptive_field": receptive_field,
                               "output_constraint": output_constraint,
                               "input_layer": current_layer,
                               "input_layer_index": network.input_layers.index(current_layer)}

            if first_call:
                # Add receptive field to cache (only for the top recursion call)
                _add_receptive_field_to_cache(model=network, receptive_field=receptive_field, start_layer=start_layer, stop_layer=stop_layer)

            return receptive_field


def calculate_receptive_field(network):
    """
    Calculate the receptive field of a given neural network.

    Args:
        network (keras.layers.Layer): Neural network to compute the receptive field for.

    Returns:
        Edge length (in pixels) of the input taken into consideration for calculating one output pixel.
    """
    receptive_field = analyze_receptive_field(network, start_layer=None)
    return receptive_field["receptive_field"]


def calculate_output_spacing(network, input_spacing):
    """
    Calculate the output spacing of a given neural network.

    Args:
        network (keras.layers.Layer): Neural network to compute the scale factor for.
        input_spacing (float or list of float): Spacing of the images used as input to the network.
                                                Can be a single scalar for single input networks or
                                                a list of scalars for multi-input networks.

    Returns:
        The spacing of the output of the network, respecting the spacing of the input.
    """

    receptive_field = analyze_receptive_field(network=network, start_layer=None)
    scale_factor = receptive_field["scale_factor"]

    if is_list_like(input_spacing):
        input_spacing = input_spacing[receptive_field["input_layer_index"]]
    return input_spacing * scale_factor


def suggest_input_sizes(network, wanted_input_size):
    """
    Takes a network and an input size and suggests several input sizes that are
    valid input sizes for the network (respecting the output constraint) and that
    are near the wanted_input_size.

    Args:
        network (keras.layers.Layer): Neural network to compute the scale factor for.
        wanted_input_size (int): Wanted input size to suggest a valid input size for.

    Returns:
        List of three tuples (input_size, output_size), each describing a valid input_size
        near the wanted_input_size.
    """
    receptive_field = analyze_receptive_field(network)
    f, c = receptive_field["output_constraint"]

    output_for_wanted_size = (wanted_input_size - receptive_field["receptive_field"]) / (1.0 * receptive_field["scale_factor"])
    rounded_output = round(output_for_wanted_size * f + c)

    def valid_input(output_size):
        """
        Calculates a valid input size for the output size
        """
        return output_size * receptive_field['scale_factor'] + receptive_field['receptive_field']

    smaller_output = (rounded_output - c - 1) / f
    equal_output = (rounded_output - c) / f
    larger_output = (rounded_output - c + 1) / f

    return [(valid_input(smaller_output), smaller_output), (valid_input(equal_output), equal_output), (valid_input(larger_output), larger_output)]


def _get_receptive_field_from_cache(model, start_layer, stop_layer):
    """
    Tries to get a receptive field from the cached stored in the model.

    Args:
        model (keras.models.Model): Neural network model to calculate receptive field of.
        start_layer (keras.layers.Layer or list of keras.layers.Layer or None): Layer to start the calculation at.
        stop_layer (keras.layers.Layer or list of keras.layers.Layer or None): Layer to stop calculation at.

    Returns:
        The receptive field for the given pair of layers, or None, if the receptive field is not cached yet.
    """
    if hasattr(model, "receptive_fields"):
        start_layer = start_layer[0] if isinstance(start_layer, list) else start_layer
        stop_layer = stop_layer[0] if isinstance(stop_layer, list) else stop_layer
        return model.receptive_fields.get((start_layer.name, stop_layer.name), None)
    return None


def _add_receptive_field_to_cache(model, receptive_field, start_layer, stop_layer):
    """
    Adds a receptive field dictionary to the cache of a model.

    Args:
        model (keras.models.Model): Neural network model to calculate receptive field of.
        receptive_field (dict): Receptive field to add to cache.
        start_layer (keras.layers.Layer or list of keras.layers.Layer or None): Layer to start the calculation at.
        stop_layer (keras.layers.Layer or list of keras.layers.Layer or None): Layer to stop calculation at.
    """
    if not hasattr(model, "receptive_fields"):
        model.receptive_fields = dict()
    start_layer = start_layer[0] if isinstance(start_layer, list) else start_layer
    stop_layer = stop_layer[0] if isinstance(stop_layer, list) else stop_layer
    model.receptive_fields[(start_layer.name, stop_layer.name)] = receptive_field
    
# -----------------------------------------------------------------------------------------------------------------------------
# Check types of layers
# -----------------------------------------------------------------------------------------------------------------------------


def layer_is_type(layer, *types_to_check):
    """
    Checks if the given lasagne.layers.Layer is an instance of one of the types given in types_to_check
    """
    return type(layer) in types_to_check


def is_conv2d(layer):
    """
    Check if layer is a 2D convolutional layer
    """
    from keras.layers import Conv2D
    return layer_is_type(layer, Conv2D)


def is_maxpool2d(layer):
    """
    Check if layer is a 2D max-pool layer
    """
    from keras.layers import MaxPool2D
    return layer_is_type(layer, MaxPool2D)


def is_deconv2d(layer):
    """
    Check if layer is a deconvolutional layer
    """
    from keras.layers import Deconv2D, Conv2DTranspose
    return layer_is_type(layer, Deconv2D, Conv2DTranspose)


def is_upsampling2d(layer):
    """
    Check if a layer is an upsampling layer.
    """
    from keras.layers import UpSampling2D
    return layer_is_type(layer, UpSampling2D)

