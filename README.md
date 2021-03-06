# Fast Guided Image Filter implemented by OpenCV 3.0 (Java API)
Discription: An implementation of "Fast Guided Image Filter" with OpenCV 3.0 (Java API). Guided image filtering is proposed in "Guided Image Filtering (ECCV 2010)" by Kaiming He, Jian Sun, and Xiaoou Tang. (http://research.microsoft.com/en-us/um/people/kahe/eccv10/index.html)
- guidance image: I (should be a color (RGB) image)
- filtering input image: p (should be a gray-scale/single channel image)
- local window radius: r
- regularization parameter: eps
- subsampling ratio: s (try s = r/4 to s=r)

## Example
```java
Mat I = Imgproc.imread("I.jpg");
Mat p = Imgproc.imread("p.jpg");
I.convertTo(I, CvType.CV_32F);
p.convertTo(p, CvType.CV_32F);
Core.divide(I, new Scalar(255.0, 255.0, 255.0), I);
Core.divide(p, new Scalar(255.0), p);

int r = 60;
double s = r / 4;
double eps = 0.000001;

FastGuidedFilter fastGuidedFilter = new FastGuidedFilter();
Mat q = fastGuidedFilter.filter(I, p, 2*r+1, eps, s, -1);

Core.multiply(q, new Scalar(255), q);
q.convertTo(q, CvType.CV_8UC1);
```
