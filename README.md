# CSLGAN

* Official repository for the paper "Automated Construction Site Layout Design System for Prefabricated Buildings using Transformer based Conditional GAN". Considering copyright issues from the engineering projects of partner companies, more information would be available on request after publication.

* ```custom_dataset.py```: arrange the custom dataset to fit the input of baseline (pix2pix)
* ```parse_label_json.py```: parse json lables that annotated from Labelme
* ```split_dataset.py```: split the dataset into train, val, and test
* ```vis_xml.py```: visualize the xml samles
* ```geatpy_solver.py```: data augmentation using NSGA-II algorithm
* ```baseline-pix2pix```: a pytorch implementation of pix2pix from [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)