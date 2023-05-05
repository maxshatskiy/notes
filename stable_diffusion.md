###### Preliminary introduction

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

###### Stable diffusion.

* CLIP embeddings
* The VAE (variational autoencoders)
* Unet
* Noise schedulers
* Transformers and self-attention

The output of the U-Net (the noise residual) is used to compute a denoised latent image representation via a scheduler 
algorithm.

**Schedulers:** PNDM, DDIM, K-LMS.

Explanation of Stable Diffusion due to Jeremy Howard:



