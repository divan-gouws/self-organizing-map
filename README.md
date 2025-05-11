# Self-organizing Map
Basic Python implementation of a self-organizing map (SOM), adapted from [AI Junkie](http://www.ai-junkie.com/ann/som/som1.html)

A preview of the fitted end result:

![image](https://github.com/user-attachments/assets/47d38144-f8fe-4e13-94e5-df36b4edefb9)

## Algorithm description
In this example, the algorithm teaches the computer to organize colors on a 2D grid by similarity, like a 2D color sorter. The following conventions will be used:
- Each color is represented by a 3D `[Red, Green, Blue]` (RGB) vector with values ranging between 0 and 1, for example `[1, 0, 0]` is used for red.
- The grid is a 2D map of small cells where each cell holds a vector. These cells will be updated to learn.
- A best matching unit (BMU) is the cell on the map that is most similar to the cell we're looking at.
- Euclidean distance is used to measure similarity between vectors.

The algorithm is composed of the following steps:
1. Initializing variables and hyperparameters.
   - Creating a 40x40 grid to place the colors.
   - Setting the input_dimension to 3, since RGB vectors are represented by 3D vectors.
   - Sigma to control the initial neighbourhood size.
   - Alpha is how fast each cell learns.
   - Creating eight colors to train on:
     `train_data = [red, green, blue, dark_green, dark_blue, yellow, orange, purple]`
2. Assign a random RGB vector to each cell in the 40x40 map. At this point the map knows nothing.
3. Pick a random color from the `train_data` vector to show to the map. For example, red.
4. Find the BMU to the chosen color by comparing it to each cell on the grid and using the Euclidean distance to determine which cell is most similar to the chosen color.
5. Shrink learning rate and neighbourhood to bring about smaller adjustments as the training process goes on.
6. Update the weights.
   - Cells near the BMU get updated to become more like the selected color.
   - The closer a cell is to the BMU, the bigger the update. This is like the map saying "Oh, this spot likes red? Let's make nearby cells a little red too."

Finally the map will show a color gradient with similar colors near to each other. 
