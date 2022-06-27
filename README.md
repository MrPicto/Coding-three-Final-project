
# 1-describe and contextualise your work
The model I used is disco GAN, a generative adversarial network model for style migration between two images. The code contains four files. x-train and y-train are used to train the model. x-test and y-test are used to train the model. discogan.py is used to train the code. model.py is mainly the model structure. test.py is used to test the model code. dataset.py is used to build the dataset.
![IMG_3158](https://user-images.githubusercontent.com/92038037/175996174-4a2ba524-61eb-459a-b251-96762bf7f48b.JPG)

# 2-link to at least one video documenting your work
[video.mp4.zip](https://github.com/MrPicto/Coding-three-Final-project/files/8994203/video.mp4.zip)

# 3-outline the design and development processes

Because I think the current machine learning applications for images in the medical field are relatively few, and the medical imaging images are interesting. And DiscoGAN is a style migration network for implementing style transfer between images. So I use this to implement medical image enhancement.
![IMG_3156](https://user-images.githubusercontent.com/92038037/175996236-5c29c283-f5a4-47a0-ade1-c62911526676.JPG)
![IMG_3157](https://user-images.githubusercontent.com/92038037/175996241-719afcf3-e706-4da6-a3f9-0f1b781f72ca.JPG)


# 4-describe the process of evaluating your project and the results of your evaluation

Evaluating the results of image enhancement has not been easy. The training curve shows that the peak curve fluctuates, but there is an overall continuous improvement. This means that the valid information retained by the enhanced image is not distorted.

# 5- the third-party resources of my project
The third party resource comes from my family, this is an unpublished dataset containing 320 coronary angiography images and 30 cerebrovascular CT images. The main purpose is to enhance another medical image by machine learning the cerebrovascular CT image style. To make the vascular structures clearer. To facilitate comparison by doctors. 
