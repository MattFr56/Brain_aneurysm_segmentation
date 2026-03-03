AI-based segmentation project on circle of Willis (CoW) aneurysms from CTA
The main idea is to design a robust CTA segmentation model based on a pixel-wise Self-supervised learnin (SSL) model (COVER :created by YutingHe) and an Attention Unet in order to :
1) Accelerate masks annotation for any downstream tasks
2) 3D printing (with some post-processing steps) real patients' aneurysm to education and training (interventional neuroradiologists) purposes.
Curating a whole set of CTA with their corresponding masks is extremely time-consuming and demand a meticulous work.
While supervised learning needs huge ammount of data wich can hinder its utilization, the combination of SSL may be benificial in figuring out the best number of data/results tradeoff. 
All the .py and .ipynb are runnable on free google colab.
