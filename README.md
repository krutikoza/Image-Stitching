# part1.py Procedure
## Steps Taken:

- Calculate ORB features using openCV library, take Keypoints and Descriptors of these features
- Calculate top two similar matches using `KnnMatch()`
- Filter only significant matches using threshold value
- Generate a similarity matrix using number of matches between two images that are below the threshold 
- Apply scaling and `PCA`
- Used KMeans Clustering
- Calculate the performance of the model
- Output the result in `output.txt` file

## Experiments:

#### Feature Matching
- We first wrote matching number of features using different distance metric such as hamming distance and euclidean distance
- White the results were decent, it was too slow to compare each image with each other
- Finally, we decided to use `BFMatcher` and `KnnMatch` to calculate the top two similar matches of features

#### Clustering
- We tried various clustering algorithms but none of them gave good engouh results
- Tried Agglomerative Clustering but result was around 0.60
- Used Spectral Clustering and the results was around 0.58
- Evaluated result on Kmeans as well but that was quite worse than Agglomerative and Spectral, around 0.51
- So we decided to convert the matrix using `StandardScalar` using sklearn, Applied `PCA` using sklearn
- Then I tried Agglomerative clustering on resultant data and the result was not good, around 0.45 which is expected
- Similarly, I tried Spectral clustering and even that performend worse as expected
- Then we tried KMeans that gave us the best results, fluctuating between **0.74** and **0.81**

## Results:

We achieved the accuracy of around **0.74** and **0.81**

In terms of Feature Matching, the result were good in some images but some were very bad. Below is the example of How some results were good and some were bad,

#### CASE 1: Correct Detection

![image](https://user-images.githubusercontent.com/61111725/210469230-3037b00d-d947-4133-a0fc-9705eef27be5.png)


#### CASE 2: Correct Detection

![image](https://user-images.githubusercontent.com/61111725/210469268-a213855c-f367-4c8f-ab74-67643bfb43a5.png)

#### CASE 3: Incorrect Detection

![image](https://user-images.githubusercontent.com/61111725/210469292-b811df40-4b45-442c-8369-1e40242a5a46.png)

#### CASE 4: Incorrect Detection

![image](https://user-images.githubusercontent.com/61111725/210469310-c5449631-1a10-4cb1-b04b-affc383544bf.png)


Though the result was quite good, the accuracy could have been improved by using more complex data for clustering other than just simple distance matrix. One trend we noticed in clustering was that a lot of images were gathering in the same cluster which might be because of the image type. More exploration is required to learn more about this behavior.

# part2.py Procedure 
## Steps taken:
- Find transformation matrix 
- Inverse warp
- Bilinear interpolation


# Find transformation matrix
- Create a transformation matrix based on option from command line
- Used a different procedure for finding transformation matrices based on the option from command line
- Reshape the transformation matrix into 3x3 matrix  
- Return the inverse of the transformation matrix


### Create transformation matrix

**n=1 (Translation)**
- For n=1, I used tx = x-x' and ty=y-y' directly instead of using numpy's solver.
- The resulting matrix looks like [[1,0,tx],[0,1,ty],[0,0,1]]
- This matrix calculates translation from destination to source, I return the inverse of this matrix to get mapping of points from source to destination.

**n=2 (Eucledian/ Rigid Transformation)**
- For n=2, I find the translation matrix as defined above
- I find the rotation matrix as multplication of 3 shear matrices
- The three shear matrices are
    - sh1: [[1, -np.tan(rad/2),0],[0,1,0],[0,0,1]]
    - sh2: [[1, 0,0],[np.sin(rad),1,0],[0,0,1]]
    - sh3: [[1, -np.tan(rad/2),0],[0,1,0],[0,0,1]]
- I then reshape each shear matrix to 3x3 shape which easies multiplication operation and multiply them 
- I multiply the resulting rotation matrix with translation matrix which returns a transformation matrix

**n=3 (Affine transform)**
- For n=3, I use system of linear equations to solve for transformation matrix
- The solution is a closed form solution and is obtained by H = A^-1.b 
- The final matrix is of form [[a,b,c],[d,e,f],[0,0,1]]

**n=3 (Projective transform)**
- For n=4, I use system of linear equations to solve for transformation matrix
- The solution is a closed form solution and is obtained by H = A^-1.b 
- The final matrix is of form [[a,b,c],[d,e,f],[g,h,1]]

# Inverse warp
- I multiply a homogenous point in an empty image [c,r,1] with inverse of the transfrom matrix to find an inverse projection point. The resulting point is of the form [x_,y_,w]
- I adjust the values for x_ and y_ coordinates with w.
- I then call bilinear interpolation on the the point (x_,y_) if it lies within the bounds of the original image and fill the pixel values at that location from original image

# Biliear Interpolation
- I take the inverse projected coordinates of the new image and then try to find pixel values from source image
- If the values of coodinates are within bounds of original image, it Biliear Interpolation finds values for that pixel 
from it's neighbouring pixels
- I used a weighted sum to calculate the final pixel values
- The weights are based on distance of the projected point from it's nearest pixels (nearest integer in x and nearest integer in y). For example if dx is distance of point from nearest integer x value and dy is the distance of point from nearest integer y value, a point at [x1,y1] can have a weight of (1-dx)*(1-dy) and a point [y2,x2] can have a weight of dx*dy


## Some environment setup guidelines that I followed
- I created a new conda environment for CV with it's environment.yml file.
- I included all packages necessary for my algorithms in this file and I create and update the environment based on the following commands

### Creating conda env
-  conda env create --file=environment.yml

### Update conda env
- conda env update --name="name of env" --file=environment.yml

# Assumptions
- I assume that the input format for part 2 would be: part2 n destination_image_path source_image_path output_image_path coordinates of destination image coordinates of source image ... coordinates of destination image coordinates of source image

# Some results from book images

## inputs 

### source image

![image](https://user-images.githubusercontent.com/61111725/210469456-187f3fd8-aac3-4e89-bb82-7c0078dff035.png)

### destination image

![image](https://user-images.githubusercontent.com/61111725/210469496-65a288c4-7515-41fd-8bcb-58ca109f701e.png)

## outputs 

### n=1 -> Translation

![image](https://user-images.githubusercontent.com/61111725/210469506-63aaeb5e-4848-4cdd-bccb-b0ab710669cf.png)

### n=2 -> Euclidean

![image](https://user-images.githubusercontent.com/61111725/210469521-980a484c-fbeb-473a-85bc-e9fc2643a2ce.png)

### n=3 -> Affine

![image](https://user-images.githubusercontent.com/61111725/210469537-3fb92656-228c-47d1-9c9d-3afc2da7d4e0.png)

### n=4 -> Projective

![image](https://user-images.githubusercontent.com/61111725/210469550-8f03a9a4-b395-48d2-984d-42c87da5952c.png)

# Results on the lincon image 

### source image

![image](https://user-images.githubusercontent.com/61111725/210469572-25f96d4e-0c27-466d-b3f1-1975fb232c66.png)

### output
![image](https://user-images.githubusercontent.com/61111725/210469584-21830ae3-39f7-40e2-b040-4b40027935ba.png)


# Sample commands
- We have added sample commands in commands.txt which might be helpful to run the code as a source file
- Prior to running the source file, we make the file executable using chmod +x a2

# part3.py

## steps taken:
- Taken points using orb.
- Applied ransac to remove outliers and got the best homograpyh matrix.
- Transformed the image using homography matrix.
- Combined the First image and transformed image.


## RANSAC Method
- We have two images as input, image1 and image2. We are transforming image2 in perspective of image1.
- ORB points of source image and destination image is taken as an input to compute ransac. At first, number of iteration is selected as 200 which gives acceptable answer. Four random pair of samples are taken from the feature points of image1 and image2. And homography matrix is calculated to transform those four points selected from image2 to perspective of image1. So we will get value of H, where H is homography matrix (image1 <- H <- image2). 
- After getting H, every point of the image2 is taken and transformed using H. Transformed point and it's pair in image1(which we got using ORB) are compared. If transformed point is in certain threshold, that point is acceptable. We will increase our count of acceptable points.

- Selecting four random point, get transformation matrix, compare every value and store maximum acceptable points and its transformation matrix.
- After repeting above steps n times(200 in our case), we will get best possible transformation matrix which is acceptable by maximum number of points.

## Transformation after RANSAC.
- After applying RANSAC, now we have transformation matrix and feature points without outliers.
- First of all, we need empty matrix which will hold our transformed image2. For that we will take four corner points and transform them using our transformation. By this, we will get where the image2 points will fall on image1 after applying transformation. Again, we are transforming image2 into prespective of image1. Using those four corner points we calculated maximum length requied for transformed image2. Same way we will calculate the dimension of our final image which is image1 + image2(transformed).
- Now we will transform our image2 using transformation matrix H. We will store that image2(transformed into perspective of image1) in top right corner of our final image. And save the image1 on the bottom left of that same final image which will result in our final panaroma.

## Assumptions
- We assumed that the image is stiched on right top of the image. In our case, image2 is stiched on right top of image1.

## Results
- Some results are stored in "Stiched_images_example" folder. Below is an example.
## Input Image 1:
![image_1](https://user-images.githubusercontent.com/61111725/219487381-9d8525ac-54cb-4b5a-b365-3e5a0f8914e8.jpg)
## Input Image 2:
![image_2](https://user-images.githubusercontent.com/61111725/219487468-3e2170a7-d63f-4832-955c-5bdbf255d16a.jpg)
## Output:
![output](https://user-images.githubusercontent.com/61111725/219487482-bd930742-3d0a-494a-b16a-b9fcf6260a65.jpg)

##  Problems
- After getting transformation matrix, while applying RANSAC, we need to calculate it's inverse. Some transformation matrix are singular matrix and no inverse is possible for those matrix which resulted in failure of our code. Four points are selected randomly so if any singular matrix is found, code was failing. It was happening totally random and surprisingly frequent. Fixed it using 'Try Except'.
## Observation 
- While testing on some images, we observed that, image stitching, running our code, requires images (image1 and image2) to be overlapped significantly. Else code might not find proper ORB points and transformation will fail.
