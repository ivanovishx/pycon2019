# pycon2019

Link del proyecto:

http://dron-aid.com 

Codigo: 

https://github.com/ivanovishx/drone_AID_Hackaton_TCDisrupt2017

Descargar para este tutorial:

#1 https://github.com/tensorflow/models

#2 https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

#3 http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

#4 https://www.dropbox.com/s/va9ob6wcucusse1/inference_graph.zip?dl=0


Tutoriales utilizados para Tensorflow:

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e 

https://modelzoo.co/model/objectdetection


Link de la presentacion:

https://docs.google.com/presentation/d/1At2aw6ncrLejHSE_drYpiIGqXMnmOOSubRgGFGAVLS8/edit?usp=sharing 


# Instrucciones


Tutorial: Train an Object Detection Classifier for mMltiple Objects using Tensorflow

Objective: Run and train an object detection model

1. Create a new enviroment in anaconda and Install tensorflow from non installed packages
2. Download all the Python packages
3. Specific additions to the PATH and PYTHONPATH variables
4. Download Tensorflow Object Detection API repository from Github
	.Create a new floder: "tensorflow1"
		-Will contain:
			-Tensorflow object detection framework
			Link: https://github.com/tensorflow/models
			-Training images
			-Training datta
			-Trained Classifier
			-Configuration files
			-And more fo the object detection Classifier
	.Download the framework from https://github.com/tensorflow/models
	.Extract it into the  "tensorflow1" folder
	.Rename models-master to models
5. Download the Faster-RCNN-Inception-V2-COCO model from Tensorflow''s model Zoo	
	(The object detection model: pre-trained classifier with specific neural network architecture)
		-Faster architecture, less acurracy: SSD-MobileNet model
		-Slower architecture but more acurracy: Faster-RCNN model
			-Download link http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
		-Unzip to the folder: \tensorflow1\models\research\object_detection 
6. Download the source code(this tutorial) repository and unzip to \tensorflow1\models\research\object_detection
	-We should already have arround 44 itens in the directory (CSV files and TFRecords) needed to train the "Pinochle Deck/Human detector" playing card Detector
	-And includes Python Scripts used to generate the training data.
	--->And Scripts to test out the object detection classifier on images, videos or webcam feed
	-To train your own "Pinochle Deck human detector" follow steps to  generate the TFRecord files
7. Download from Dropbox the frozen interface graph to work out of the box https://www.dropbox.com/s/va9ob6wcucusse1/inference_graph.zip?dl=0
	-Extract the content to \object_detection\inference_graph
	-Test it runnig Object_detection_image.py (or video or webcam) script
		-To train your own object detector:
			-delete All files in \object_detection\images\train and \object_detection\images\test
			-delete The “test_labels.csv” and “train_labels.csv” files in \object_detection\images
			-delete All files in \object_detection\training
			-delete All files in \object_detection\inference_graph
			-is ready to start from scratch in training your own object detector.
			-this tutorial go on to explain how to generate the files for your own training dataset
8. Set up new Anaconda Virtual enviroment
	-Open terminal in Anaconda and create a new virtual enviroment called "tensorflow1":
		// $conda create -n tensorflow1 pip python=3.5
		// $activate tensorflow1
	-Install tensorflow-gpu in this environment by issuing:
		// $(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow-gpu
	-Install the other necessary packages by issuing the following commands:
		// (tensorflow1) C:\> conda install -c anaconda protobuf
		// (tensorflow1) C:\> pip install pillow
		// (tensorflow1) C:\> pip install lxml
		// (tensorflow1) C:\> pip install Cython
		// (tensorflow1) C:\> pip install jupyter
		// (tensorflow1) C:\> pip install matplotlib
		// (tensorflow1) C:\> pip install pandas
		// (tensorflow1) C:\> pip install opencv-python
		NOTE:  ‘pandas’ and ‘opencv-python’ are not necesary for Tensorflow but are using by python scrips to generate TFRecord and work with images, video, webcams
9. Configure PYTHONPATH enviroment variables
	We must to create a PYTHONPATH that points to the directories	
	\models
	\models\research
	\models\research\slim
	On terminal (tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
	-MAC:
	$ vim ~/.bash_profile 
	(insert press "i")
		export PYTHONPATH="/Users⁩/ivanlozano⁩/workHome⁩/tensorflow⁩/aidDroneTensorflow⁩/tensorflow1⁩/models"
		export PYTHONPATH="/Users⁩/ivanlozano⁩/workHome⁩/tensorflow⁩/aidDroneTensorflow⁩/tensorflow1⁩/models/research"
		export PYTHONPATH="/Users⁩/ivanlozano⁩/workHome⁩/tensorflow⁩/aidDroneTensorflow⁩/tensorflow1⁩/models/research/slim"
		export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
	(Note: Every time the "tensorflow1" virtual environment is exited, the PYTHONPATH variable is reset and needs to be set up again)
10.	Compile Protobufs and run setup.py
	Protobuf files are used by TensorFlow to configure model and training parameters
	-Installation tutorial: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
		// (may be diferent for Windows, see point 2f on tutorial)
	-This creates a name_pb2.py file from every name.proto file in the \object_detection\protos folder.
	-Run the following commands from the C:\tensorflow1\models\research directory:
		(tensorflow1) C:\tensorflow1\models\research> python setup.py build
		(tensorflow1) C:\tensorflow1\models\research> python setup.py install
11. Test TensorFlowsetup to verify it works
	-Launch the object_detection_tutorial.ipynb script with Jupiter
	-From the \object_detection directory, issue this command:
		(tensorflow1) C:\tensorflow1\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
	This opens the script in your default web browser and allows you to step through the code one section at a time.
		(Note: part of the script downloads the ssd_mobilenet_v1 model from GitHub, which is about 74MB. This means it will take some time to complete the section, so be patient.)
		Once you have stepped all the way through the script, you should see two labeled images at the bottom section the page. If you see this, then everything is working properly! 
12. Train the new detection classifier:
	Now that the TensorFlow Object Detection API is all set up and ready to go, we need to provide the images it will use to train a new detection classifier.
	-You can use your phone to take pictures of the objects
	-or download images of the objects from Google Image Search
	-At least 200 pictures overall. Make sure the images aren’t too large. They should be less than 200KB each, and their resolution shouldn’t be more than 720x1280
	After you have all the pictures you need, move 20% of them to the \object_detection\images\test directory, and 80% of them to the \object_detection\images\train directory
13. Label pictures:
	LabelImg is a great tool for labeling images, and its GitHub page 
	Download and install LabelImg, point it to your \images\train directory, and then draw a box around each object in each image. Repeat the process for all the images in the \images\test directory. This will take a while!
	GitHub: https://github.com/tzutalin/labelImg 
	Download: https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1
	-LabelImg saves a .xml file containing the label data for each image. These .xml files will be used to generate TFRecords, which are one of the inputs to the TensorFlow trainer. Once you have labeled and saved each image, there will be one .xml file for each image in the \test and \train directories
14. Generate Training Data
15. Create Label Map and Configure Training
16. Run the training
17. Export Inference Graph
18. Use Your Newly Trained Object Detection Classifier!


Flow of the procedure:

-->Select the model to train the object detection clasifier

	-->Train the object detection clasifier(we will use SSD because of this Mac)
	
	-->Setup the TensorFlow Object Detection API to use pre-trained models for object detection
	
		-->Train a new detection classifier (#12)
		
			-->Label Pictures
			
				-->Generate Training Data
				


OK Instructions::

-->Create the tensorflow1 working space

	-->Activate Tensorflow
	
		-->Install all the dependences
		
			-->set the PYTHONPATH enviroment variables
			
				-->Unzip the democode from the tutorial
				
					-->Unzip the Trained model for object detection(from model Zoo)
					
						aka: "inference graph"
						-->In the "object_detection" directory run the script 
						"Object_detection_webcam.py"







LinkS:

-Tutorial  https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
-Detection multiple objects https://www.youtube.com/watch?v=COlbP62-B-U
