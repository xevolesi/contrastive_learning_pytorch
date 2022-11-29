# contrastive_learning_pytorch
Implementation of paper "Learning a Similarity Metric Discriminatively, with Application to Face Verification" by Chopra et. al.

# Installation steps
1. Clone the repo and change dir to it;
2. You need to initialize fresh environment. The simplest way to do this is
just to use conda:

        conda create -n contrastive python=3.9 -y
        conda activate contrastive

3. You need to install dependencies:

        pip install -r requirements.txt

   If you want to develop this repo further the install dev-dependencies by
   running:

        pip install -r requirements.dev.txt