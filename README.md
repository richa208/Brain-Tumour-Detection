# Brain-Tumour-Detection
A project for detecting brain tumour through MRI images

The code has been tested on BRATS 2018 dataset having High-Grade Gliomas (HGG) and Low Grade Gliomas (LGG) cases.
Every patient has multimodal MRI data in the dataset and has four MRI scanning sequences :
- T1 - weighted (T1)
- T1 with gadolinium enhanced contrast (T1c)
- T2-weighted (T2) and,
- Fluid Attenuated Inversion Recovery (FLAIR).

In addition to these sequences, ground truth segmentations with the following 4 intra-tumoral
classes are also provided -
- Necrosis
- Edema
- Non-enhancing
- Enhancing tumor
