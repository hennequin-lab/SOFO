# SOFO:Second-order forward-mode optimization of recurrent neural networks for neuroscience

## Abstract
Training recurrent neural networks (RNNs) to perform neuroscience tasks can be challenging. Unlike in machine learning where any architectural modification of an RNN (e.g.\ GRU or LSTM) is acceptable if it facilitates training, the RNN models trained as _models of brain dynamics_ are subject to plausibility constraints that fundamentally exclude the usual machine learning hacks. The “vanilla” RNNs commonly used in computational neuroscience find themselves plagued by ill-conditioned loss surfaces that complicate training and significantly hinder our capacity to investigate the brain dynamics underlying complex tasks. Moreover, some tasks may require very long time horizons which backpropagation cannot handle given typical GPU memory limits. Here, we develop SOFO, a second-order optimizer that efficiently navigates loss surfaces whilst _not_ requiring backpropagation. By relying instead on easily parallelized batched forward-mode differentiation, SOFO enjoys constant memory cost in time. Moreover, unlike most second-order optimizers which involve inherently sequential operations, SOFO's effective use of GPU parallelism yields a per-iteration wallclock time essentially on par with first-order gradient-based optimizers. We show vastly superior performance compared to Adam on a number of RNN tasks, including a difficult double-reaching motor task and the learning of an adaptive Kalman filter algorithm trained over a long horizon.

## Installation steps

- install opam : https://opam.ocaml.org/doc/Install.html
- choose opam switch 5.0.0 
- install dune (run `opam install dune`) 
- install owl (run `opam install owl`); you will need to have OpenBLAS installed (see below in case issues with OpenBLAS arise). 
- install base, torch, owl_ode, bos (`opam install base torch owl-ode bos ppx_accessor ppx_deriving_yojson`)
- clone https://github.com/hennequin-lab/cmdargs and do `dune build @install` followed by `dune install` (after `cd`-ing into the corresponding directory)

## To run examples

- compile the example by running e.g. `dune build src/example.exe`. If linking issues arise, please get in touch.
- run `dune exec _build/default/src/example.exe -d [results_directory]` (where `[results_directory]` is where you want your results to be saved). Depending on the example file you are trying to execute, there might be additional command line arguments.
 
## OpenBLAS installation

- on certain operating systems linking errors to OpenBLAS can arise when installing owl. One solution to circumvent them is to install OpenBLAS from source (https://github.com/xianyi/OpenBLAS.git), and to then manually include the path to the OpenBLAS installation in LD_LIBRARY_PATH and PKG_CONFIG_PATH.
