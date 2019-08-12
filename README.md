# PointwiseNet
## Pointwise Geometric and Semantic Learning Network on 3D Point Clouds
The geometric and semantic information of 3D point clouds significantly influence the analysis of 3D point cloud structures. However, semantic learning of 3D point clouds based on deep learning is challenging due to the naturally unordered data structure. In this work, we strive to impart machines with the knowledge of 3D object shapes, thereby enabling machines to infer the high-level semantic information of the 3D model. Inspired by the vector of locally aggregated descriptors, we propose indirectly describing the high-level semantic information through the relationship of each point's low-level geometric descriptor with a few visual words. Based on this approach, an end-to-end network is designed for 3D shape analysis, which combines pointwise low-level geometric and high-level semantic information. A spatial transform and uniform operation are applied in the network to make it invariant to input rotation and translation, respectively. In addition, our network also employs pointwise feature extraction and pooling operations to solve the unordered point cloud. In a series of experiments with popular 3D shape analysis benchmarks, our network exhibits competitive performance in many important tasks, such as 3D object classification, 3D object part segmentation, semantic segmentation in scenes, and commercial 3D CAD model retrieval.


**Our main contributions include the following:**
- We present a novel method based on the VLAD mechanism to extract high-level semantic information of point clouds, which is indirectly described by the relationship of each point's low-level geometric descriptor with a few visual words, such that machines can infer the high-level semantic information of the 3D point cloud.
- We propose an end-to-end network for 3D shape analysis, named PointwiseNet, which combines pointwise low-level geometric and high-level semantic features. The network has the ability to classify and segment 3D models and does not require any pretraining.
- Four different strategies, including pointwise feature extraction, spatial transform, uniform operation, and pooling operation, enable PointwiseNet to address the rotation invariance, translation invariance, and disorder of point cloud data.		
- The effectiveness of the presented network is validated on a number of benchmark datasets,  demonstrating encouraging improvements. Some extensive experiments are also conducted to verify our claims and justify our design choices in PointwiseNet.


## The network architecture
<img src="https://github.com/djzgroup/PointwiseNet/blob/master/img/flowchart.jpg" width="600">

The network architecture of our PointwiseNet consists of three main components: pointwise feature learning, classification and segmentation.
- *Pointwise feature learning.* 
The pointwise feature learning consists of three phases:  STN,  KNN, and  VLAD. (a) The STN module is used to apply transformations such as rotation and translation. (b) The KNN module is used to extract the pointwise low-level geometric information for each point of the 3D point cloud. (3) The VLAD module is used to extract the pointwise high-level semantic information for each point of the 3D point cloud, which is indirectly described by the relationship of each point's low-level geometric descriptor with a few visual words.
- *Classification network.*
For the 3D object classification task, the complete classification network consists of pointwise feature learning and classification. 
The feature learning takes N points as input, applies input and feature transformations, and then aggregates the pointwise features into a global feature vector by a global pooling layer. After the global pooling layer in the first component, the 3D point cloud is represented as a 1024-dimensional feature vector. To classify the point clouds,  three fully connected layers are attached after the global feature vector. The final output is C scores for all C candidate classes.
- *Segmentation network.* 
For the 3D semantic segmentation, the complete segmentation network consists of pointwise feature learning and segmentation. 
The feature learning takes a single object for part region segmentation as input, while the segmentation component concatenates the three output vectors (low-level geometry vector, high-level semantic vector, and global feature vector) into a 1536-dimensional feature vector and then inputs it into four fully connected layers to obtain the final classification result, which is N*M scores for each of the N points and each of the M semantic subcategories.


## How does our VLAD module works?
We show how to leverage the VLAD mechanism to extract the high-level semantic features from the 3D point set. VLAD is a popular descriptor pooling method for both instance-level retrieval and image classification. Arandjelovic et al. proposed an end-to-end deep network named NetVLAD that stores the sum of residuals for each visual word (cluster centre) of a 2D image and performs image-based retrieval for place recognition. PointNetVLAD leverages on the success of PointNet and NetVLAD to perform 3D point-cloud based retrieval for large-scale place recognition.

The pointwise high-level semantic feature (e.g., skeleton or part of the 3D model) is an implicit expression that is difficult to describe directly. Inspired by PointNetVLAD and NetVLAD, we can indirectly describe the high-level semantic feature by the relationship between each point’s low-level geometric descriptor and a few visual words.\
[Click on the hyperlink for details.](https://github.com/djzgroup/PointwiseNet/blob/master/HowDoesOurVladModuleWorks.pdf)


## Overview code directory
${ROOT}/\
   ├── image/\
   ├── models/ :contains data preparation, model definition.\
   ├── part_seg/ :includes training scripts for 3D object part segmentation.\
   ├── sem_seg/ :includes training scripts for Semantic segmentation in scenes.\
   ├── utils/ :contains some utility functions.\
   ├── train.py/ : training scripts for 3D object classification.\
   ├── evaluate.py\
   ├── README.md


## Best model
**You can download the best trained model to verify our results.**
- 3D object classification(*modelnet10*)\
  [[1024,94.5]](https://drive.google.com/file/d/1K2RGAm5KQ4WYSRaPpfF_u12b7B4ieyAc/view?usp=sharing)  [[2048,95.0]](https://drive.google.com/file/d/1CqYDabSQ6XFx9wDyutDud10aC7g5YtCy/view?usp=sharing)   [[5000,95.1]](https://drive.google.com/file/d/1CIcz9rzkE7cLyPQ1DPpK_8nZsvrApKJX/view?usp=sharing) 
- 3D object classification(*modelnet40*)\
  [[1024,91.3]](https://drive.google.com/file/d/1wIZB83qaEUAA8gmXUGZXHlAIVCkiAWiw/view?usp=sharing)  [[2048,91.6]](https://drive.google.com/file/d/1wK7JorE9Q2Wh4BwfJZjbEsQLBQf3YKp_/view?usp=sharing)   [[5000,92.7]](https://drive.google.com/file/d/17-zHO7V99iFjzE_dFXsAKxPaYAJ_AIh9/view?usp=sharing) 
- 3D object part segmentation(*ShapeNet*)\
  [[85.1]](https://drive.google.com/file/d/1q_sBZtbJoygv6gFm35Bq2Kvetr1XXELp/view?usp=sharing) 
- Semantic segmentation in scenes(*S3DIS*)\
  [[4096,83.36]](https://drive.google.com/file/d/1q2rsqdtLi2EXxvEQvG6Uom0wESD2KB6u/view?usp=sharing) 


## Results
**Hypothesis testing**\
Taking the 3D object classification experiment as an example, we prove that the proposed method is significantly different from other methods through the hypothesis testing.\
[Click on the hyperlink for details.](https://github.com/djzgroup/PointwiseNet/blob/master/HypothesisTesting.pdf)

**3D object part segmentation**\
Results of part segmentation on the validation data of the ShapeNet part dataset. Best viewed in color. Examples are plane, bag, cap, car, chair, earphone, guitar, knife, lamp, laptop, motor, mug, pistol, rocket, skate and table. Due to space limitations, it is impossible to show all the examples; thus, we randomly selected a model from each category for visual comparison. For each group of objects, the leftmost one is the ground truth, the middle one was predicted by PointNet, and the right one was predicted by PointwiseNet.

<img src="https://github.com/djzgroup/PointwiseNet/blob/master/img/part_seg.jpg" width="600">

**Semantic segmentation in scenes**\
Qualitative results for semantic segmentation. From left to right: original input scenes; ground truth point cloud segmentation; PointNet segmentation results and PointwiseNet segmentation results. Best viewed in color. We selected 5 room scenes (from top to bottom are conference room #1, office #1, office #3, lounge #1, and lobby #1) from the evaluation dataset for display. The first column is the input point cloud, with the walls and ceiling hidden for clarity.  The second, third, and last columns are the ground truth segmentation, the prediction from PointNet, and the prediction from PointwiseNet, respectively, where the points belonging to different semantic regions are coloured differently (chairs in red, tables in purple, bookcase in green, floors in blue, clutters in black, beam in yellow, board in grey, and doors in khaki).

<img src="https://github.com/djzgroup/PointwiseNet/blob/master/img/sem_seg.jpg" width="600">

**Commercial 3D CAD model retrieval**\
The retrieval results. Left column: queries. Right five columns: retrieved models from the 3D CAD model database.

<img src="https://github.com/djzgroup/PointwiseNet/blob/master/img/retrival.jpg" width="450">


## Complexity Analysis
The following table summarizes the space (number of parameters) and the time (floating point operations) complexity of PointwiseNet in 3D object classification task with 1024 points as the input. Compared with PointNet++ [40], PointwiseNet reduces the parameters by 4.7% and the FLOPs by 52.0%, which shows its great potential for real-time applications, e.g., scene parsing in autonomous driving.

Method 	| #params | #FLOPs (Inference) 
-|-|-
PointNet [1]      |3.48M	 |14.70B
PointNet++ [2]    |1.48M	 |26.94B
3DmFVNet [3]      |45.77M	 |16.89B
PointwiseNet      |1.41M	 |12.92B


## References
- [1] Qi C R, Su H, Mo K, et al. Pointnet: Deep learning on point sets for 3d classification and segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 652-660.
- [2] Qi C R, Yi L, Su H, et al. Pointnet++: Deep hierarchical feature learning on point sets in a metric space[C]//Advances in neural information processing systems. 2017: 5099-5108.
- [3] Ben-Shabat Y, Lindenbaum M, Fischer A. 3dmfv: Three-dimensional point cloud classification in real-time using convolutional neural networks[J]. IEEE Robotics and Automation Letters, 2018, 3(4): 3145-3152.
- [4] Li J, Chen B M, Hee Lee G. So-net: Self-organizing network for point cloud analysis[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 9397-9406.

## Acknowledgment
This work was supported in part by the National Natural Science Foundation of China under Grant 61702350 and Grant 61472289 and in part by the Open Project Program of the State Key Laboratory of Digital Manufacturing Equipment and Technology, HUST, under Grant DMETKF2017016.
