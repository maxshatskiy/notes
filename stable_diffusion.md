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

**VAE** is optional, but saves a lot of time and money since it reduced dimention based on which model for 
noise prediction is trained.

**Change**: make input to NN pixels+one hot encoded version of the original image, this model should be better 
at predicting noise, since it knows what was actual input.
Now if we feed actual image of the digit, then the model makes prediction of the noise. In this way we give guidance
of what the image we are trying to create.

**How to do to create input for "cute teddy"?** How to turn this text to embedding?

We need a model, which takes "cute teddy" and outputs representation on how it could be expressed.
To train such a model we need an image and a text tag.
Based on this we can create: 
1. text --> textencoder --> random vector, if the model was not trained
2. image --> imageencoder --> random vector, if the model was not trained
we would like that vector ouputs of both encoders for similar objects are similar, we want dot product
should be big. and the text and image, which are not similar should have a small dot product.
Then we can add this dot product of those which should be similar and substract for those which should be not similar
then we can train text and image encoders, which give text and image embeddings. Now text and image are in the same 
features space. This model ise CLIP, which is trained with Contrastive Loss. 

CLIP test encoder takes text and outputs embeddings. Similar text gives similar embeddings.

The outputs of text encoder now can be used to train a network which is used to predict noise.

Text-->text encoder--> embeddings (similar to embeddings of the images)-->train noise prediction network
Now Unet can be guided by text captions.

**Score function** - gradients of the loss with respect to pixel values.

**Time stamps**. we used various level of noise. it is possible to create a noise schedule - a monotonically decreasing
function f(t)-which returns the amount of noise. This is a way to pick of how much noise to use.
We could think of simply of jow much noise to use - this is defined by beta.
We randomly pick a minibatch, we randomly pick a image and randomly pick a t and then pick noise.

**Inference.** Model starts with noise, then we multiplied prediction of the noise and multiplied by a constant,
then subtracted from a noisy image and got a deionised image, which is closer to the better image. Then we have to
repeat this process. Which C to use? How to substract the noise? The C is analogous to LR and therefore concepts
of "momentum" and "adam" can be used.

All diffusion models, take image with noise, original image and time stamp t (if we think about this approach as 
in differential equations).

# Latents

General diffusion models are machine learning systems that are trained to denoise random Gaussian noise step by step
to get to a sample of interest such as an image. Hugging face diffusers: 
https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb

Diffusion models operates on pixel space and therefore slow --> latent diffusion operate in lower dimensional space
and the model is trained to represent a compressed representation of image.

# Papers

## Distillation
### Progressive Distillation for Fast Sampling of Diffusion Models

We take a teacher - slow and big NN, and teach a students, which is faster.
The reason why original diffusion process gets significant number of steps.
We train a new model, which takes as input and intermediate image of diffusion process, and compare it with final
model, which we want to get.
Teacher model is a complete Stable Diffusion model.
Noise-->Step 1 --> Step 2, then we train student model, which can run directly noise-->2, then this model becomes a 
teacher.

### On Distillation of Guided Diffusion Models
Classifier-free guided diffusion models (CFGD). We put prompt and an empty prompt. Then we get puppy and empty prompt
then we get average and process further with diffusion.
CFGD - without prompt.


Student model gets noise, prompt, guidance.

### Imagic: text-based real image editing with diffusion models
Modifying part of the picture with a prompt. Dog can be modified to "as sitting dog", etc.
1. Start with Stable Diffusion model
2. Text--> text encoder--> Diffusion model
3. Fine tune embedding --> to output as input image, which was provided as reference
4. Fine-tune entire model--> text embeddings are fixed
5. Target embedding+slightly modified embedding into fine-tuned model and we get modified photo.

## Pipeline
1. CLIP tokenizer - split to units. everything should be the same length, therefore padding is used
2. CLIP encoder - take input token and get embeddings. For classifier free guidance we need embeddings for empty string.
3. Prompt embeddings and empty string embeddings are concatenated.
4. VAE
5. UNET

Scaling of random noise - depending on the stage we have to scale noise and keep variance in control.
Timestampas are not integers and this number defines amount of noise applied.
??? Guidance Scale -???









