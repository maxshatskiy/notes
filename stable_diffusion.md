# Preliminary introduction

**Classifier-free guidance** - is a method to increase the adherence of the output to the 
conditioning signal (in our case text). Details about guidance for generative models: https://sander.ai/2022/05/26/guidance.html

**Negative prompt**  - another prompt and scaling the difference between of that prompt and 
the one used for conditional image. Basically we try to subtract negative prompt from the 
conditional image.

**Other uses:** instead of starting with pure noise, we can start with some image and continue 
diffusion process from it. Further we can take a generated image and use it as a starting 
image for next diffusion process, for example to change style to Van Gogh.
**Strength** defines to which extent to use the original image and to which extend to use 
conditional prompt.

**Guidance scale** - one version with a prompt and one without a prompt and then takes an average 
of the two pictures.

**Fine-tuning.** It is possible to fine tune diffusion model for generating particular images.
Get images--> use image captioning model to generate captions for those images-->then fine-tune 
diffusion model using image-caption pairs.
Use this fine-tuned model to output images.

**Textual inversion** - a special kind of fine-tuning, where we fine-tune just a single embedding.
We can "teach" a new word to the text model and train its embeddings close to some visual representation.
Add new token to vocabulary-->freeze weight of all models, but not of text encoder-->train on visual representative images.

**Dreambooth**  - fine-tuning that attempts to introduce new subject by providing jsut a few images of the new subject.
Use existing token and bring to the pictures we required. For example, we can encode word "sks" to be used for my
picture and then ask model to draw it in a certain style. We ask a model to overfit the prompt to certain images.
Training script: https://github.com/huggingface/diffusers/tree/main/examples/dreambooth

**Prior preservation** - technique to valid overfitting by performing special regularization using other examples of the term 
in addition to those, which were provided.

# Stable diffusion

* CLIP embeddings
* The VAE (variational autoencoders)
* Unet
* Noise schedulers
* Transformers and self-attention

The output of the U-Net (the noise residual) is used to compute a denoised latent image representation via a scheduler 
algorithm.

**Schedulers:** PNDM, DDIM, K-LMS.

**Explanation of Stable Diffusion due to Jeremy Howard.**

Stable diffusion for handwritten digits.
image(x1) -->black-box model (f)--> probability that x1 is handwritten digit p(x1)=0.98
image(x2) -->black-box model (f)--> probability that x2 is handwritten digit p(x2)=0.4
image(x3) -->black-box model (f)--> probability that x3 is handwritten digit p(x3)=0.02
If we have this black-box model (f) then we can generate a new handwritten digits.

We change on pixel of an image -->black-box model (f)--> probability changes, for example increases
We can change every pixel one by one and see how it changes the probability of the changed picture beeing handwritten

We found a gradient of probability with respect to change of every pixel value.
Instead of changing the weights we change pixels of the input image by a gradient, which we obtained and
obtain a new image, which is more likely looks like a handwritten digit.

Any arbitrary noisy input to a desired picture.

Finite-differencing of calculating derivative is very slow.

**The problem, we do not have a function f, which provides a probabilities.**
We need a NN, which can decide which pixels to change to get a picture, which looks as hands written digit.

We can take a TRUE handwritten digit and add some noise, then some more noise, then more and 
then predict probabilities that those figures are handwritten digits.

Or we can cast the problem in another way: we can predict noise, which was added to original image.

The amount of noise says us "how much is the figure is a digit".
Here we think about NN as a box having INPUT, OUTPUT and LOSS function and derivative is used to update weight.
INPUTS are images with different amount of noise, OUTPUTS: the amount of noise 
at the image (parameters of the mean and variance), or we can predict not amount of noise, but actual noise itself.

_IF we can predict the noise, then we get a "derivative" or the answer to the question "by how much" should we
change the pixel values to bring the image to the one, which has higher probability of being an image of handwritten
image._

We take normal image and add different amount of noise and train NN to predict amount of noise.

## Prediction of noise
**NOW we can generate images:** pure noise-->pass to NN--> which part of the input is noise--> how to change 
every pixel to make and input picture look more like a digit.

The NN, which is used for medical image segmentation: **Unet - the first component of Stable Diffusion.**

**Input to the Unet:** somewhat noisy image, it could be pure noise or image without any noise.
**Output of the Unet:** is the noise, such that if we substract this noise from input image, 
then we have noise free image.

**Problem**
image of 512x512x3 colored image has 786432 pixel and therefore training such model will take a lot of time.
How to train the model more efficiently?

##_**Autoencoder = VAE**_

We know it is possible to compress pictures. If to put picture to a Convolutional layer with stride 2 with 6 channels,
then we put again and again to a Conv layer with expanding number of channels we can compress it down to 64x64x4 = 16384
pixels. We can also get the image back using inverse convolution (deconvolution) and restore original image.

If we put a picture to a model, we initially get a random noise, but with loss of MSE, the model tries to get the 
image inside the model and restore the original image.

Encoder part can be used to compress the image. Having decoder we can restore the compressed image.

The intermediate layer between encoder and decoder contains all the information about image. All 10mln pictures
are put into encoder and then are feed into nnet unit. It does not take noisy image, but it takes "latent"
representation of the image. Network predicts noise, which can be substracted from noisy latent, which gives 
actual latents. Then the latents are output to the decoder, which outputs a large image.

**VAE** is optional, but saves a lot of time and money.



