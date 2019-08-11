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

## HOW DOES OUR VLAD MODULE WORKS
[xxxxxxxxx](https://github.com/djzgroup/PointwiseNet/blob/master/HowDoesOurVladModuleWorks.pdf)

## Results
**3D object part segmentation**
Results of part segmentation on the validation data of the ShapeNet part dataset. Best viewed in color. Examples are plane, bag, cap, car, chair, earphone, guitar, knife, lamp, laptop, motor, mug, pistol, rocket, skate and table. Due to space limitations, it is impossible to show all the examples; thus, we randomly selected a model from each category for visual comparison. For each group of objects, the leftmost one is the ground truth, the middle one was predicted by PointNet, and the right one was predicted by PointwiseNet.
<img src="https://github.com/djzgroup/PointwiseNet/blob/master/img/part_seg.jpg" width="600">

**Semantic segmentation in scenes**
Qualitative results for semantic segmentation. From left to right: original input scenes; ground truth point cloud segmentation; PointNet segmentation results and PointwiseNet segmentation results. Best viewed in color. We selected 5 room scenes (from top to bottom are conference room #1, office #1, office #3, lounge #1, and lobby #1) from the evaluation dataset for display. The first column is the input point cloud, with the walls and ceiling hidden for clarity.  The second, third, and last columns are the ground truth segmentation, the prediction from PointNet, and the prediction from PointwiseNet, respectively, where the points belonging to different semantic regions are coloured differently (chairs in red, tables in purple, bookcase in green, floors in blue, clutters in black, beam in yellow, board in grey, and doors in khaki).
<img src="https://github.com/djzgroup/PointwiseNet/blob/master/img/sem_seg.jpg" width="600">

**Commercial 3D CAD model retrieval**
The retrieval results. Left column: queries. Right five columns: retrieved models from the 3D CAD model database.

<img src="https://github.com/djzgroup/PointwiseNet/blob/master/img/retrival.jpg" width="450">

## HOW DOES OUR VLAD MODULE WORKS


## Acknowledgment
This work was supported in part by the National Natural Science Foundation of China under Grant 61702350 and Grant 61472289 and in part by the Open Project Program of the State Key Laboratory of Digital Manufacturing Equipment and Technology, HUST, under Grant DMETKF2017016.
