# Sequential Attend, Infer, Repeat: Generative Modelling of Moving Objects

This is an official Tensorflow implementation of Sequential Attend, Infer, Repeat (SQAIR), as presented in the following paper:
[A. R. Kosiorek](http://akosiorek.github.io/), [H. Kim](https://hyunjik11.github.io/), I. Posner, Y. W. Teh, [Sequential Attend, Infer, Repeat: Generative Modelling of Moving Objects](https://arxiv.org/abs/1806.01794).

* **Author**: Adam R. Kosiorek, Oxford Robotics Institute & Department of Statistics, University of Oxford
* **Email**: adamk(at)robots.ox.ac.uk
* **Webpage**: http://akosiorek.github.io/

## Dependencies
Install [Tensorflow v1.6](https://www.tensorflow.org/versions/r1.6/install/),  and the following dependencies
 (using `pip install -r requirements.txt` (preferred) or `pip install [package]`):
 * numpy==1.14.2
 * matplotlib==2.1.1
 * dm_sonnet==1.14
 * attrdict==2.0.0
 * scipy==0.18.1
 * orderedattrdict==1.5

## Sample Results

SQAIR learns to reconstruct a sequence of imaes by detecting objects in every frame and then propagating them to the following frames. This results in unsupervised object detection & tracking, which we can see in the figure below. The figure was generated from a model trained for 1M iterations. The maximum number of objects in a frame (and therefore number of detected and propagated objects) is set to four, but there are never more than two objects. The first row shows inputs to the model (time flies from left to right), while the second row shows reconstructions with marked glimpse locations. Colors of the bounding boxes correspond to object id. Here, the color is always the same, which means that objects are properly tracked.

![SQAIR results](https://raw.githubusercontent.com/akosiorek/sqair/master/resources/sqair_mnist/000037.png)

![SQAIR results](https://raw.githubusercontent.com/akosiorek/sqair/master/resources/sqair_mnist/000050.png)

![SQAIR results](https://raw.githubusercontent.com/akosiorek/sqair/master/resources/sqair_mnist/000098.png)

The model here was trained on sequences of up to 10 time-steps. We have also evaluated on 100 time-step sequences, where objects' motion is much more noisy than in the training data. [Here are the results.](https://youtu.be/vIVaK6LK-qE)

## Data  
Run `./scripts/create_multi_mnist_dataset.sh`
The script creates train and validation datasets of sequences of multiple moving MNIST digits.

## Training
Run `./scripts/train_multi_mnist.sh`
The training script will run for 1M iterations and will save model checkpoints every 100k iterations and training progress figures every 10k iterations in `results/multi_mnist`. Tensorflow summaries are also stored in the same folder and Tensorboard can be used for monitoring. The model is trained with a curriculum of sequences of increasing length, starting from three time-steps and increasing by one time-step every 100k iterations to the maximum of 10 time-steps. The process can take several days on a GPU.


## Experimentation
The jupyter notebook available at `notebooks/play.ipynb` can be used for experimentation. It is set up to load a model pre-trained for 1M iterations. It is from a different run than the results report in the paper or presented above and its performance is slightly worse. You can download more model checkpoints by running `./scripts/download_models.sh`.

## Tinkering with the model
SQAIR is fairly sensitive to hyperparameters controlling weight between different terms in the loss: standard deviation of the output distribution and biases added to statistics of prior distributions of discovery and propagation. Hyperparameters we chose for MNSIT have generalised well to the DukeMTMC dataset, but we have noticed that they need tweaking when using other datasets. For example, with current values of hyperparameters, SQAIR does break if you run it on a moving MNIST dataset with digits 30% smaller to what we use. In case you would like to run SQAIR on your own dataset, we recommend tinkering with the following hyperparameters found either in `sqair/common_model_flags.py` or in the model config at `sqair/configs/mlp_mnist_model.py`.

```
transform_var_bias, output_scale, scale_prior, prop_prior_step_bias, output_std, disc_step_bias, prop_step_bias
```

These  hyperparameters are documented in their corresponding config files. The last three (`output_std, disc_step_bias, prop_step_bias`) are probably the most important ones.


## Citation

If you find this repo or the corresponding paper useful in your research, please consider citing:

    @inproceedings{Kosiorek2018sqair,
      title={Sequential Attend, Infer, Repeat: Generative Modelling of Moving Objects},
      author={Kosiorek, Adam Roman and Kim, Hyunjik and Posner, Ingmar and Teh, Yee Whye},
      booktitle={Advances in Neural Information Processing Systems},
      url = {https://arxiv.org/abs/1806.01794},
      pdf = {https://arxiv.org/pdf/1806.01794.pdf},
      year={2018}
    }


## License

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

## Release Notes
**Version 1.0**
* Original implementation; contains the multi-digit MNIST experiment.
