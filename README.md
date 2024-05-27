# SpecIV/SpecPCL
Spectral Representation for Causal Estimation with Hidden Confounders

## Requirements
Please see the `requirements.txt` file for the required packages. 


## Directory
- `datasets/` contains the datasets we will use in the experiments.
	- `demand_design.py` generates the synthetic demand design dataset.
	- `dsprite.py` generates the synthetic dSprites dataset.
- `algos/` contains the implementation of the algorithms.
    - `embedding_sgd.py` implements the dual embedding SGD algorithm, i.e., the vanilla version of our method.
	- `spectral_repr.py` implements the Spectral Representation for Causal Inference algorithm.
- `configs` contains the configuration files for the experiments. In each yaml file:
- `networks/` contains the implementation of the neural network parametrizations used in the experiments.
    - `image_models.py` implements the CNN for feature extraction.
	- `contrastive_models.py` implements the MLP for the contrastive loss-based model (CTRL).
- `utils/` contains the implementation of the utility functions.
    - `dist.py` implements the distance functions.
	- `lin_alg.py` implements the linear algebra functions.
    - `kernel.py` implements the kernel constructing functions.
    - `sample_generator.py` implements the sample generating functions used in the embedding SGD experiments.
- `logs/` contains the logs of the experiments. 
- `main.py` is the entrance of the algorithm.


## Run the code
To run the code, you can use the following command:
```
python main.py --config <path-to-configs>
```


