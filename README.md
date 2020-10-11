# Neural_Networks
---

### Resources
---
###### 
TensorFlow 2.2, Keras, Python, Deep Learning, Neural Networks, scikit, pandas

### Overview
---
###### 
Alphabet Soup requests model that will predict the success of a venture based on data they provided from past ventures.  A binary claification model will be used to predict future results.

### Analysis
---
###### 
I started with a 2-layer deep learning model to evaluate the dataset; this was activated with a Rectified Linear Unit (RELU) function.  The number of neurons was set to twice the number of features in the scaled dataset.  Then I started consolidating those inputs in the second layer to drive the model towards the binary output faster by cutting them in half.  I also experimented by reducing the number of neurons by an order of magnitude in the second layer; this increased predictive accuracy faster, with a total increase of 0.5%.  Increasing the number of epochs had no or a negative effect (slight decrease changing 50 to 100 epochs; 500 & 1000 epochs started bringing the accuracy score closer to the first run of 50).

I also implemented a different model; I used a Sigmoid activation function.  This function has log regression as its foundation which can fit binary outputs well, as in this case when seeking a yes or no investment recommendation.  The results were not significantly different than the RELU model.

### Results
--- 
###### 
The highest accuracy achieved was 72.9% by using 3 neural layers.  The first neural layer was set up as a 2x multiple of the number features; compressing the number of neurons in the second layer increased accuracy fractionally in every variation, regardless of activation functions (Sigmoid or Rectified Linear Unit), and expanding the third layer had better results than decreasing the number of neurons in the third layer relative to the second layer or keeping them the same.  Bottom-line, none of the model perturbations achieve the target of 75% accuracy.

### Recommendations
---
###### 
For the current data set, a two layer model, either RELU or Sigmoid is sufficient, with between 50-100 epochs.  As noted in the data cleansing commentary of the code, there are problematic issues with the data categorization design.  I have encountered this elsewhere, and when categories are so broad as to encompass more than 50% of the data set, their contribution is little while their computation time is high.  That is to say it has little business utility.  Recommend further parsing broad parent categories into discrete child units.  This will improve any learning model.
