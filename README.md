<h1>Touch Code</h1>

<h3>Overview</h3>

This project focuses on segmenting and processing Braille characters from images using YOLOv8 for detection and CNN-based classification for recognition. Instead of predicting the Braille characters directly, YOLOv8 is utilized solely for detecting and extracting character coordinates, enabling precise cropping of the input image.

<h4>Workflow</h4>
<b>Image Segmentation & Cropping</b>
<br><br>
The input image is processed using YOLOv8, which detects individual Braille characters and provides their bounding box coordinates.
The detected characters are cropped and saved locally, with filenames dynamically assigned based on their position in the image.
The sorting follows a structured format, starting from the top-left corner and proceeding left to right, then moving line by line downward. <br>
<br><br>
<b>Character Recognition</b>
<br><br>
Each cropped Braille segment is passed through a CNN-based prediction model trained on a dataset of 600–800 images per character.
The model predicts the corresponding Braille character, forming the basis of the final translation.
<br><br>
<b>Text Reconstruction & Formatting</b>
<br><br>
The recognized characters are compiled into an array and sorted based on spacing between detected characters.
Spaces and line breaks are determined based on the relative distances between cropped segments.
The final translated text is reconstructed and formatted for output.
Features <br>
✅ Automated Braille Character Detection via YOLOv8 <br>
✅ CNN-Based Character Prediction for accurate recognition <br>
✅ Dynamic Image Cropping & Sorting based on spatial positioning <br>
✅ Space & Line Detection for structured text formatting <br><br>

This project provides an efficient pipeline for extracting and translating Braille text from images, making it a valuable tool for accessibility and assistive technology applications.
