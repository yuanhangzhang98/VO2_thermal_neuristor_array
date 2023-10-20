# VO~2~-based spiking oscillators
This is a PyTorch model for simulating VO~2~-based spiking oscillators (termed "thermal neuristors"). Description of the design and underlying principles can be found in the paper [Reconfigurable Cascaded Thermal Neuristors for Neuromorphic Computing](https://onlinelibrary.wiley.com/doi/epdf/10.1002/adma.202306818).

In the original paper, only two neuristors are simulated. In this model, the parallel implementation on GPU using the PyTorch library makes it possible to efficiently simulate thousands of neuristors simultaneously, enabling us to efficiently explore the collective dynamics for a large number of neuristors. New paper on this topic under preparation. 
