the poject overview:
1. Door detection AI models: yolov8m, yolov8n , rf-detr, faster r-cnn
- Label formats: for door_detection_dataset: yolo labels, for roboflow_dataset: yolov8 format.
- All training codes train the models with the yolo formatted labels
- for rf-detr: there is a code to change yolo format to coco format
2. auto label code: uses a roboflow model to label the doors
3. door and preview code: give a preview of 50% of the images labeled
4. finetune: use the best model of the initial training and train them on roboflow dataset (has door, hinge, knob, and lever)
5. door sim: inwardly open door, enters only, gazebo, simulated the rover with lidar and camera
6. door sim commit mark: returns too, no encoders needed, reverses steps from where it committed to entering the building
7. inward: door open inwardly, goes in and spins 360 degrees and comes out, same steps as door sim
8. outward: door open outward, same as inward
9. rover real deployment: to be used in ros2 jazzy, and eloquent in jetson nano 2gb developer kit, no encoders needed, accesses decision node, door detection node, camera and lidar nodes, ai model, motion controller and motor controller
10. The 3d print models made in CAD: tank bottom, and tank top.
