# generative-boson-sampling

#### Author: Sean Huver
#### Email: huvers @ gmail 

## Background 
Boson-Sampling is a fancy sounding physics experiment that may be thought of as a pachinko (or plinko for you Price is Right fans) parlor game for photons, as well as being a non-universal quantum computer (we'll get to this in a bit). One has a number of input modes <i>**m**</i> into which a number of photons <i>**n**</i> are inserted and then interact with <i><b>(m*(m-1)/2)</b></i> beam splitters, which in the pachinko analogy are the pins the marbles bounce off of and may go in one direction or another. In Boson-Sampling, we'd like to discover the probability distribution for where the <i>**n**</i> photons may wind up at the output modes.

<img src="https://i.imgur.com/A3npDj2.jpg">

The challenging feature of Boson-Sampling, as pointed out by creators Scott Aaranson and Alex Arkhipov <b>[1]</b>, is that calculating the probability distributions of various outcomes of the experiment becomes classically intractable for large values of **m** and **n**. This is because the Hilbert space of the experiment grows due to photon path-entanglement as:

<img src="http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7Bm%21%7D%7Bn%21%28m%20-%20n%29%21%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\frac{m!}{n!(m - n)!} " width="89" height="46" align="center"/>

As well as the fact that Bosons have complex probability amplitudes in their interactions with beam splitters. It was shown by Scheel <b>[2]</b> that boson probability amplitudes are actually related to matrix <a href="https://en.wikipedia.org/wiki/Permanent_(mathematics)">permanents</a>, a problem known to be <b>#P-complete</b>.

This is why Boson Sampling is interesting -- It is a simple experiment where a quantum system can do something a classical system cannot (even if there is no currently known killer application for doing so). 

## generative-boson-sampling 

The goal of this project is to explore the ways in which Deep Learning may or may not be useful for efficiently (and accurately!) exploring the properties of Boson-Sampling. In particular, we begin with using the <a href="https://github.com/XanaduAI/strawberryfields"> Strawberry Fields library from Xanadu </a> to create suitable machine learning training sets <a href="https://github.com/huvers/generative-boson-sampling/blob/master/src/boson_sampling_data_generator.ipynb"> here.</a> Our aim is to create a generalized format for quickly generating data from various boson sampling configurations which will then be used to train generative ML models with Tensorflow.

The first experiment is to train a <a href="https://github.com/huvers/generative-boson-sampling/blob/master/src/nn_bsampler_solver.ipynb">relatively simple densely connected neural network</a> to predict various Boson-Sampling configurations and their possible outcomes, and to do so faster than can be calculated with Strawberry Fields in a brute force appraoch.

<img src="https://i.imgur.com/025ICn3.png">

<img src="https://i.imgur.com/PcPW7KU.png">

We then turn to creating <a href="https://github.com/huvers/generative-boson-sampling/blob/master/src/autoencoder_boson_sampling.ipynb">Autoencoder models to do the same</a>.

Lastly, we create <a href="https://github.com/huvers/generative-boson-sampling/blob/master/src/BS-GAN.ipynb">Boson-Sampling GANs that generate their own initial configurations and then determine what their probability distributions look like</a>.

<b> Note:</b> This is very much a work in progress :)

## References

[1] S. Aaronson and A. Arkhipov. The computational complexity of linear optics. Theory of Computing, 9 (4):143–252, 2013.

[2] Stefan Scheel. Permanents in linear optical networks. 2004. quant-ph/0508189.
