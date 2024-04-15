# Melody Reduction Algorithm

Melody reduction can be seen as an abstraction of a melody. 
This code repository offers a simple computational approximation, i.e., Melody Reduction Algorithm, via shortest-path finding.
The algorithm is previously named "tonal reduction algorithm" to emphasize the preferred "linear motion" and "arpeggiation" in melody development under a harmony context. However, such implementation can be confusing. For example, our algorithm can work for music that is not tonal. To avoid confusion, it is renamed to melody reduction algorithm.

This algorithm is used in the paper:
> Ziyu Wang, Lejun Min, and Gus Xia. Whole-Song Hierarchical Generation of Symbolic Music Using Cascaded Diffusion Models. ICLR 2024.


## Use Case
Although most music reduction study (e.g., GTTM) is based on a notated score (e.g., A# and Bb are encoded differently), 
this algorithm takes in MIDI-like format as input. The algorithm works for:
1. A melody with two extra annotations: 1) underlying chord progression and 2) phrase label.
2. Chord resolution is one beat at most.
3. Time signature can be 3/4 or 4/4.
4. The algorithm is tested on pop music. But can be tested on other genres.

## Limitations
1. The current implementation does not allow time signature change.
2. The hyperparameters of TRA are not well-tuned. 
3. The plot function is not well-developed.

## Use the algorithm
Please check the functions in `run_melody_reduction.py`. 

To get the demo files, run

```
python example_reduction_pop909.py
```

The [POP909 dataset](https://github.com/music-x-lab/POP909-Dataset) is the version downloaded and cleaned from the [repo](https://github.com/Dsqvival/hierarchical-structure-analysis).

