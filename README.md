# Scalable Vision-Guided Crop Yield Estimation (AAAI 2026 Oral)

[Paper](https://www.arxiv.org/abs/2511.12999) | [Dataset](https://doi.org/10.5281/zenodo.17626117)

This is the official repository for the AAAI 2026 oral paper *"Scalable Vision-Guided Crop Yield Estimation."*

There are two main folders in this code appendix.

The `computer_vision_model/` directory contains the Python code used for training the computer vision models on our dataset.

The `ppi/` directory contains R code for post-processing the photo model predictions using our proposed PPI++ methodology, including the field-level observations we are able to release from Nigeria and Zimbabwe, along with the yield predictions for each of these fields from our trained computer vision models. We also provide the code to reproduce all figures and tables in the main text and technical appendix.

Each folder has a separate README file with more details.
