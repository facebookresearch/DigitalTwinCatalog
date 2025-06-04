# PBIR
This stage depends on two main libraries, `psdr-jit`, an unbiased differentiable pathtracer, and `irtk`, a toolkit that facilitates inverse rendering experiment setup. They are at experimental stage now and will be officially released later. 

This stage uses [`gin-config`](https://github.com/google/gin-config) for configuration. It's important to familiarize it.

If you would like to access the source code for `irtk`, you can find its location via
```bash
python -c 'import irtk; print(irtk.__file__)'
```

## Walk Through

### `run.py`
In `run.py`, we define the high level pipeline for this stage within `pipeline()`， which simply runs substages define in `configroot`. Currently, we have three substages:
* `microfacet_naive-envmap_sg`: optimize the material naively and optimize a set of spherical Gaussains as the environment map. 
* `microfacet_basis-envmap_ls`: optimize the material using basis and optimize the environment map using the large step optimizer.  
* `microfacet_basis-envmap_ls-shape_ls`: same as the second stage but also optimize the shape using the large step optimizer. 

For more details on the components that make up these stages, read the `models` section below.

Example config files can be found in `configs/template`. When you want to modify the configurations, it is best to make them on a copy of the template directory.

### models
The `models` directory contains a list of models, each of which implements an algorithm for optimizing a certain scene component. Each substage above is merely a composition of these models. 

Currently available models are:
* `MicrofacetNaive`: Optimize a spatially varying microfacet BRDF naively, that is, optimize the texels directly.
* `MicrofacetBasis`: Optimize a microfacet BRDF using a set of BRDF basis.
* `EnvmapSG`: Optimize an environment map represented using a set of spherical Gaussian lobes.
* `EnvmapLS`: Optimize an environment map using the [largestep](https://github.com/rgl-epfl/large-steps-pytorch) optimizer.
* `ShapeLS`: Optimize an mesh using the [largestep](https://github.com/rgl-epfl/large-steps-pytorch) optimizer.

### `opt.py`
`opt.py` implements the main optimization loop and it is usually used as a stage in pipeline. Its most important input is the `model_class`, which is an `irtk.Model` that defines an optimization process of some scene parameters. 

For instance, take a look at `configs/template/microfacet_naive-envmap_sg.gin`:
```
optimize.model_class = @MultiOpt
...

MultiOpt.model_classes = [
    @MicrofacetNaive,
    @EnvmapSG,
]

`MultiOpt` is defined in `irtk.model` and it's used to wrap multiple `irtk.Model` together, in this case, ``
MicrofacetNaive.mat_id = 'mat'
MicrofacetNaive.s_max = 0.04
MicrofacetNaive.s_min = 0.04
MicrofacetNaive.r_min = 0.1
MicrofacetNaive.d_lr = 5e-3
MicrofacetNaive.r_lr = 1e-3

EnvmapSG.emitter_id = 'envmap'
EnvmapSG.numLgtSGs = 128
EnvmapSG.num_init_iter = 100
EnvmapSG.optimizer_kwargs = {
    'lr': 1e-3
}
...
```
`MultiOpt` is defined in `irtk.model` and it's used to wrap multiple `irtk.Model` together, in this case, ``
`MultiOpt` is defined in `irtk.model` and it's used to wrap multiple `irtk.Model` together,  `MicrofacetNaive` and `EnvmapSG` in this case.
