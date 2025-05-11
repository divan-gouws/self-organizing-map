# self-organizing-map
Basic Python implementation of a self-organizing map (SOM), adapted from [AI Junkie](http://www.ai-junkie.com/ann/som/som1.html)

A preview of the fitted end result:

![image](https://github.com/user-attachments/assets/47d38144-f8fe-4e13-94e5-df36b4edefb9)

## Algorithm description
In this example, the algorithm teaches the computer to organize colors on a 2D grid by similarity, like a 2D color sorter. The following conventions will be used:
- Each color is represented by a 3D `[R, G, B]` vector with values ranging between 0 and 1, for example `[1, 0, 0]` is used for red.
- The grid is a 2D map of small cells where each cell holds a vector. These cells will be updated to learn.
- A best matching unit (BMU) is the cell on the map that is most similar to the cell we're looking at.
- Euclidean distance is used to measure similarity between vectors.

The algorithm is composed of the following steps:
1. Initializing variables and hyperparameters
   - creating a 40x40 grid to place the colors
   - setting the input_dimension to 3, since RGB vectors are represented by 3D vectors
   - sigma to control the initial neighbourhood size
   - alpha is how fast each cell learns
   - creating eight colors to train on i.e. `train_data = [red, green, blue, dark_green, dark_blue, yellow, orange, purple]`
