# Emotion Classifier

## Description
I created this project to explore various architectures aimed at image classification.

## Models
* Basic CNN
* Basic CNN with Batch Norm
* Deep CNN
* Fully Convolutional Network
* VGG16 Transfer Learning Network

** To see model details run ```python model	_and_train.py --arch <NAME> --summary```

#### Results
- Deep CNN model had highest test accuracy of 65%
- VGG16 model reached accuracy of 55%

## Getting Started
1. Download [data](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)  from Kaggle competition and unzip into 'data/'
2. Install python dependencies with ```pip install -r requirements.txt```
3. Run ```python model_and_train.py --help``` to see available options
4. Try ```python model_and_train.py --arch basic --summary``` to print the Keras architecture layout of 'basic CNN' architecture
5. Train with ```python model_and_train.py --arch <NAME> --run_name <TEXT>```
6. Model training is logged by default so you can view it in Tensorboard: ```tensorboard --logdir=log```

#### CLI Options
	$ python model_and_train.py --help

	Usage: model_and_train.py [OPTIONS]

	Options:
	  --arch [basic|basic_with_batch_norm|deep_cnn|fully_conv|vgg]
			                  model architecture to train with
	  --summary                       PRINT model summary for architecture
	  --save_dir TEXT                 directory to log to
	  --run_name TEXT                 name of run for Tensorboard
	  --lr FLOAT                      learning rate for training  [default: 0.001]
	  --dropout_rate FLOAT            dropout rate for training  [default: 0.3]
	  --num_epochs INTEGER            number of epochs to train for  [default:
			                  1000]
	  --batch_size INTEGER            batch size for training  [default: 32]
	  --patience INTEGER              Early stopping training patience  [default:
			                  10]
	  --validation_split FLOAT        [default: 0.1]
	  --test_size FLOAT               percentage of date to use for
			                  testing/validation  [default: 0.2]
	  --reduce_lr                     Reduce learning rate when val loss has
			                  stopped improving
	  --help                          Show this message and exit.

#### Transfer Learning Model
NOTE: Running the VGG model for the first time will create the bottleneck features from VGG. This will take **several minutes** to compute and use a lot of system resources.

### Dependencies
* Keras
* Click
* Tensorflow
* Numpy
* Sklearn
* OpenCV

#### TODO
- Add progress bar for events
- Add CLI image classify method
- Add data augmentation to images
- Experiment with different methods of using gray images instead of OpenCV gray2color conversion
- Add a deep residual network