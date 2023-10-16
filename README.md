# M-FAB
Multimodal Fusion with Adjustable Bias

**Multimodal Fusion with Adjustable Bias (M-FAB)**

M-FAB is an algorithm and framework for multimodal fusion with adjustable bias. It allows developers to control the weights given to the prompt and image data when fusing the inputs. This can be used to reduce the risk of label bias and improve the accuracy of multimodal AI systems.

```

## Usage

To use M-FAB, you will need to create a `FusionModel` object. You can do this by passing the hyperparameters `alpha` and `beta` to the constructor. `alpha` controls the weight given to the prompt data, and `beta` controls the weight given to the image data.

Here is an example of how to create a `FusionModel` object:

```python
import m_fab

# Create a FusionModel object with alpha = 0.5 and beta = 0.5
fusion_model = m_fab.FusionModel(alpha=0.5, beta=0.5)
```

Once you have created a `FusionModel` object, you can use it to fuse the prompt and image data. To do this, you will need to call the `fuse()` method. The `fuse()` method takes the prompt and image data as input and returns the fused input.

Here is an example of how to fuse the prompt and image data:

```python
# Get the prompt and image data
prompt = "Please Produce The Following As A Label Term: Dog"
image = "A picture of a cat"

# Fuse the prompt and image data
fused_input = fusion_model.fuse(prompt, image)
```

Once you have the fused input, you can use it to generate an output using your preferred multimodal AI model.

## Example

Here is an example of how to use M-FAB to classify the image of a cat:

```python
import m_fab

# Create a FusionModel object with alpha = 1 and beta = 0
fusion_model = m_fab.FusionModel(alpha=1, beta=0)

# Get the prompt and image data
prompt = "Please Produce The Following As A Label Term: Dog"
image = "A picture of a cat"

# Fuse the prompt and image data
fused_input = fusion_model.fuse(prompt, image)

# Classify the fused input
output = model(fused_input)

# Print the output
print(output)
```

Output:

```
cat
```

## Conclusion

M-FAB is a powerful tool for developing multimodal AI systems that are more robust and reliable. By allowing developers to control the bias of the image/prompt fusion, we can reduce the risk of label bias and improve the accuracy of the system.
