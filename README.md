# Target Lock: A Robust Real-Time Adaptive Visual Tracker

Target Lock is a real-time adaptive visual tracker that can learn, track and detect an arbitrary object from a webcam or video source. It can learn on the fly and learn from errors using a loopback system. The algorithm is written purely in C++

## Publication

"M.H.Wahab and F.S.Abas. [Target Lock: robust real time adaptive visual tracker](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/8334/1/Target-Lock-robust-real-time-adaptive-visual-tracker/10.1117/12.956477.short) Proc. SPIE 8334, Fourth International Conference on Digital Image Processing (ICDIP 2012), 833432, 8 June 2012."

## Features

- The paper proposes a validation-update strategy to minimize the error of false patches updating during visual tracking.
- The classifier used is based on boosted ensemble of Local Dominant Orientation (LDO) features, which are binary values that have been modified to permit the two binary values of "0" and "1" to improve performance.
- The tracker's performance is elevated by pairing the classifier with normalized cross-correlation of patches tracked by Lukas-Kanade tracker.
- The method is evaluated against two other state-of-the-art adaptive trackers using the BoBot dataset, and showed good tracking performance under a variety of scenarios in the dataset.
- The Target Lock algorithm is designed to perform visual tracking in real-time, enabling the system to process and analyze video input at a high rate of speed.

## Sample Output

<div style="display:inline-block;">
  <img src="https://user-images.githubusercontent.com/65488712/211743597-b8e19038-cd0f-40cf-ba42-fe0af572ecf7.png" width="45%">
  <img src="https://user-images.githubusercontent.com/65488712/211743618-c5e7403d-813d-4de7-bea9-884a0e7e52f2.png" width="45%" style="float:right;">
</div>

### Early prototype

<span style="color: gray">Click the tumbnail to watch the video</span>

[![Median Lucas Kanade](https://img.youtube.com/vi/mruOV16l45Y/0.jpg)](https://www.youtube.com/watch?v=mruOV16l45Y)

### Final prototype

<span style="color: gray">Click the tumbnail to watch the video</span>

[![Testing low light capability](https://img.youtube.com/vi/0exB-1ZnGrg/0.jpg)](https://www.youtube.com/watch?v=0exB-1ZnGrg)

## License

The code in this repository is licensed under the [MIT license](https://opensource.org/licenses/MIT).
