# Intel® Computer Vision SDK (Intel® CV SDK) | Hello World Tutorial


## Prerequisite checklist
- [ ] Verify hardware compatibility
- [ ] Install OpenCL® and other dependencies
- [ ] Install the Intel® CV SDK and set environment variables

## Install the Hello World tutorial support files (trained models, videos, images)

#### 1. Create the directory /opt/intel/tutorials/cvsdk

	mkdir -p /opt/intel/tutorials/cvsdk

#### 2. Go to the new directory.

	cd /opt/intel/tutorials/cvsdk

#### 3. Download and clone the tutorial content to the current directory (opt/intel/tutorials/cvsdk). Four sub-directories are created: models, samples, videos, and images.

	git clone https://github.com/intel-iot-devkit/computer-vision-hello-world.git


# Part 1: Optimizing and deploying a deep learning model for pedestrian detection (~15 minutes)


## Introduction and learning goals

#### 1. What you’re about to learn and why it is important.

This tutorial uses a Single Shot MultiBox Detector (SSD) on a trained GoogleNet* model to walk you through the basic steps of using the Deep Learning Deployment Toolkit’s Inference Engine, included in the Intel® CV SDK. Although this tutorial uses GoogleNet, on your own you can perform inference on other neural network architectures, such as AlexNet*. 

You will begin this tutorial by using the Model Optimizer to convert the included trained model to two Intermediate Representation (IR) files. Your result will be one .xml file and one .bin file, required by the Inference Engine. 

You will then use Inference on the Intermediate Representation files to infer meaning from the model data. Inference is the process of using a trained neural network to infer meaning from data, such as images. The code sample in this tutorial uses images by feeding a short video, frame by frame, to the Inference Engine (the trained neural network). The result is image classification output -- text on a screen that displays information about the image, and a video that shows each pedestrian surrounded by a box.

This tuturial is important to your education about deep learning and the Intel® CV SDK because of the real-life ways in which you can expand on this tutorial scenerio, and because this tutorial prepares you for more difficult deep learning scenarios. For example, this tutorial provides the basis of code for security functions like:

- Triggering an action if inference detects an individual entering an environment. For example, a homeowner might want to be alerted if a person walks onto their property, but not an animal.
- Comparing the number of people leaving an environment to the number of people entering the environment. For example, a store might want to be alerted at closing time that someone might still be on the premises.

In addition, once you complete this tutorial, you will be able expand on your knowledge of inference to identify other items, such as animals or vehicles, to satisfy other use cases.


#### 2. Provide an example of what success looks like in this module.

> Show a side-by-side comparison of the pedestrian video clip: original vs with bounding boxes
	
#### 3. Conceptual diagram: End-to-end video application developer journey map.

> (continue to show this throughout the module as it changes? i.e. ‘you are here’)

## Use the Model Optimizer to optimize a deep learning model

#### 1. Go to the sample application directory.

	cd /opt/intel/tutorials/cvsdk/samples/ped_detection

#### 2. Run the Model Optimizer on the trained Caffe* model.

This step generates one .xml file and one .bin file, both in the directory  <current_dir>/artifacts

	```sudo su```
	```source /opt/intel/computer_vision_sdk_2017.1.163/bin/setupvars.sh```
	```python runMO.py -w SSD_GoogleNetV2_caffe/SSD_GoogleNetV2.caffemodel -d SSD_GoogleNetV2_caffe/SSD_GoogleNetV2_Deploy.prototxt```

> The Model Optimizer converts a trained Caffe model to be compatible with the Inference Engine and optimizes it for Intel
architecture.
> Include the .xml and .bin files with with your C++ application to apply inference to visual data.
> Note: If you train or update the Caffe model after this point, you must run the Model Optimizer again.

#### 3. Exit super user mode.
	
	exit

Note: At several places in the tutorials, you will be asked to enter and exit super user mode. For security purposes, it is best to exit super user mode when it is not needed.

## Build the sample application

#### 1. Open the sample application source code to view the lines which call the Inference Engine.

> Explain how the application is calling the inference engine and setting parameters to define what they want to do. Give an example of how this can be changed to achieve different results and mention the related tutorial to learn more.

#### 2. Close the source file. 

#### 3. Build the sample application.

 	make

## Run the pedestrian detection sample application to use the Inference Engine on a video
*Explain the parameters prior to running the app*

	./IEobjectdetection -i opt/intel/tutorials/cvsdk/videos/vtest.avi -fr 200 -m artifacts/VGG_VOC0712_SSD_300x300_deploy/VGG_VOC0712_SSD_300x300_deploy.xml -d CPU -l pascal_voc_classes.txt

> You see a video of pedestrians with a red bounding box around each individual. You also see textual console output about the objects identified and the confidence level that the Inference Engine correctly identified and drew bounding boxes around pedestrians, versus other objects in the video. The higher the confidence level, the more likely the Inference Engine correctly identified and drew bounding boxes around the pedestrians. For example: 0.83 is more confident than 0.23.

## \(Optional) Run the pedestrian detection sample application again, using different parameters to see how they affect the results

#### 1. *define suggested activities and describe what should happen/what it means*

## Part 1 recap

In this tutorial, you:
- Used the Model Optimizer on a pre-trained Caffe model to create two IR files that are optimized and compatible with Intel architecture. 
- Identified the lines in sample code that call the Inference Engine.
- Built the sample code to create the application that will use the IR files to identify the objects (pedestrians)
- Ran the pedestrian sample application on the video to see how inference put bounding boxes around each individual.

You are ready for the next tutorial!


# Part 2: Re-purpose a Pedestrian Detection application to identify cars (~10 minutes).


## Introduction and learning goals

#### 1. Introduction and overview of what you’re about to learn and why you would care.

This tutorial will walk you through the basics of using the Deep Learning Deployment Toolkit’s Inference Engine (included in the Intel® Computer Vision SDK). Here, inference is the process of using a trained neural network to infer meaning from data (e.g., images). In the code sample that follows, a video (frame by frame) is fed to the Inference Engine (our trained neural network) which then outputs a result (classification of an image). Inference can be done using various neural network architectures (AlexNet*, GoogleNet*, etc.). 

This example uses a Single Shot MultiBox Detector (SSD) on GoogleNet model. The Inference Engine requires that the model be converted to IR (Intermediate Representation) files. This tutorial will walk you through the basics taking an existing model (GoogleNet) and converting it to IR (Intermediate Representation) files using the Model Optimizer.
	
#### 2. Provide an example of what success looks like in this module.

> Show a side-by-side comparison of the pedestrian video clip: original vs with bounding boxes
	
#### 3. Conceptual diagram: End-to-end video application developer journey map

> (continue to show this throughout the module as it changes? i.e. ‘you are here’)

## Update the pedestrian detection application code to target cars instead of people

#### 1. Navigate to the sample application directory (cd opt/intel/tutorials/cvsdk/samples).
#### 2. Create a copy of the ped_detection folder and rename it car_detection (cp -r ped_detection car_
detection)
#### 3. Open the object detection sample app source code to view the lines which call the IR.
#### 4. *David to define the steps to update the app*
#### 5. Save and close the source file.
#### 6. Build the sample application.
#### 7. Run the updated sample application to call the Inference Engine.

./IEobjectdetection -i opt/intel/tutorials/cvsdk/videos/vtest.avi -fr 200 -m artifacts/VGG_VOC0712_SSD_300x300_deploy/VGG_
VOC0712_SSD_300x300_deploy.xml -d CPU -l pascal_voc_classes.txt

## \(Optional) Explore using different parameters to see how they affect the results

#### 1. *define some suggested activities here and describe what should happen/what it means*

## Part 2 recap

#### 1. *Provide a quick recap of what you did and why you care*
#### 2. *link to the next logical tutorial(s) in the learning path*

## Additional resources

- [CV SDK landing page (IDZ)](https://software.intel.com/en-us/computer-vision-sdk?cid=sem43700020075377675&intel_term=computer+vision+sdk&gclid=CjwKCAiA9f7QBRBpEiwApLGUit1KXgtbu46anzhcsxJVBltKW-JOxPzucCmBxVDZwI_1H4FYgQZ-3RoC96sQAvD_BwE&gclsrc=aw.ds)
- [Developer guide](https://software.intel.com/en-us/cvsdk-inference-engine-apiref)
- [API references](https://software.intel.com/en-us/inference-engine-devguide)

OpenCL is a trademark of Apple Inc. used by permission by Khronos
Intel is a registered trademark of Intel Corporation
* Other brands and names may be the property of others
