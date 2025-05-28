# DATA SORTING INSTRUCTIONS
A sort instruction on how to do data sorting for Eglinton.

1. Run the script using either command line or Visual Studio Code: ``data_preprocessing_eglinton_linux_v4_flex_cam.py``
- To select the designated folder, change the variable in the `line 503` of the code as follow:
``` python
#self.cul_de_sac_filtered_Path=join(self.Selected_Sorted_Data_Stage1_Path,"cul-de-sac_filtered")#1
#self.intersection_filtered_Path=join(self.Selected_Sorted_Data_Stage1_Path,"intersection_filtered")#2
#self.main_Path=join(self.Selected_Sorted_Data_Stage1_Path,"main_filtered")#3
#self.parallel_parking_filtered_Path=join(self.Selected_Sorted_Data_Stage1_Path,"parallel_parking_filtered")#4
#self.right_hand_roundabout_filtered_Path=join(self.Selected_Sorted_Data_Stage1_Path,"right_hand_roundabout_filtered")#5
#self.small_carpark_filtered_Path=join(self.Selected_Sorted_Data_Stage1_Path,"small_carpark_filtered")#6
#self.standard_carpark_filtered_Path=join(self.Selected_Sorted_Data_Stage1_Path,"standard_carpark_filtered")#7
#self.standard_roundabout_filtered_Path=join(self.Selected_Sorted_Data_Stage1_Path,"standard_roundabout_filtered")#8

self.Raw_Data_Path=self.[Insert-folder name based on the above.]
#Example: choosing Cul-de-sac folder:
self.Raw_Data_Path=self.cul_de_sac_filtered_Path
```
2. A window should be appeared displaying two images representing front and rear camera as follow:
![A screenshot of the data-sorting Window](window.png)
3. Using the following keys to manipulate the data frames:

|Keys|Description|
|:--:|-----------|
|`W`|Increase the frame step j by 1.|
|`S`|Decrease the frame step j by 1.|
|`D`|Next images of +j steps.|
|`A`|Last images of -j steps.|
|`Esc`|Move to the next Folder. If no folder is found, the program will exit.|
|`Esc` then `Space`|Exit the program. **NOTE**: Please make sure to press the buttons follow the order.|

4. To sort a certain number of frames onto one behaviour:
    - Select the current image frame index by pressing button ``K``. (i_begin should be changed to the current index from `None`)
    - Moving onto the next set of frames following Step 3 until desired index. The desired index should be the last frame representing one behaviour of the bus.
    - Select the current image frame index as end index by pressing button ``L``.

5. Once the index range has been determined, sort the data by pressing the keys corresponding to the behaviour.

``NOTE``: Any number of frames that are evaluated as `Bad Behaviour` should be moved to the ``Bad Behaviour`` folder by pressing button ``K``.

## Keys Table
|Key|Description|
|:-:|-----------|
|1|Change the step j to 1.
|2|Change the step j to 20.
|``Y``|Lane Following Path|
|``U``|Lane Bay Pass Path|
|``I``|Pulling Path|
|``O``|Pullout Path|
|``P``|Carpark Pass Path|
|``[``|Startpoint In Path|
|``]``|Startpoint Out Path|
|``F``|Roundabout Turn Around To Beach Path|
|``G``|Roundabout Straight Path|
|``H``|Roundabout Turn Around To Office Path|
|``;``|Shed In Path|
|``'``|Shed Out Path|
|``B``|Carpark Dual Steering Path|
|``N``|Carpark Entry Path|
|``C``|Lane Bay Pass Path first-half Path|
|``V``|Lane Bay Pass Path second-half Path|
|``,``|Intersection Lane Following Path|
|``.``|Intersection Turn Around To Beach Path|
|``/``|Intersection Turn Around To Office Path|
|``Z``|Cul-de-sac Dual Steering Path|
|``X``|Others Path|

### Driving Mode

|Mode|Description|
|:-:|---|
|2|Lane Following Neural Network|
|1|GPS|
|0|Manual|
