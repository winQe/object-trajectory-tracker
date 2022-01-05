#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include <thread>
#include <mutex>
#include <deque>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "mynteyed/camera.h"
#include "mynteyed/utils.h"

#include "CDNeuralNet.hpp"


using namespace std;
using namespace MYNTEYE_NAMESPACE;


// The pre-trained neural-network for people detection:
const char* nn_weightfile = "/home/adin/object-tracker/darknet/yolov4.weights";
const char* nn_cfgfile = "/home/adin/object-tracker/darknet/cfg/yolov4.cfg";
const char* nn_meta_file = "/home/adin/object-tracker/darknet/cfg/coco.data";


double gettimeofday_as_double() {
	struct timeval tv;
	gettimeofday(&tv, NULL);

	double ts = tv.tv_sec + (tv.tv_usec / 1000000.0);

	return ts;
}

cv::Mat imRGB;

void worker_thread(CDNeuralNet _cdNet) {
	while (true) {
		if (imRGB.cols > 10) {
			_cdNet.detect(imRGB);
		} else {
			usleep(5000);
		}
	}
}

int main(int argc, char* argv[]) {

	struct timeval tv;
	int _DEPTH_WIDTH = 1280;
	int _DEPTH_HEIGHT = 720;
	int _FPS = 30;
	int _RGB_WIDTH = 1280;
	int _RGB_HEIGHT = 720;

	float _MAX_DISPLAY_DISTANCE_IN_METERS = 3.0;
	float _YELLOW_DISTANCE_IN_METERS = 2.0;
	float _RED_DISTANCE_IN_METERS = 1.0;
	deque <cv::Point> trajectory;


	// //Initialize ROS Node
	// ros::init(argc, argv, "object_detection");
	// ros::NodeHandle n;

	// ---------------------------------------------------------
	//  BEGIN:  Start MYNT-EYE-D
	// ---------------------------------------------------------
	Camera cam;
	DeviceInfo dev_info;
	if (!util::select(cam, &dev_info)) {
		return 1;
	}
	
	util::print_stream_infos(cam, dev_info.index);

	cout << "Open device: " << dev_info.index << ", " << dev_info.name << endl << endl;

	OpenParams params(dev_info.index);
	{
		params.framerate = 30;
		params.stream_mode = StreamMode::STREAM_1280x720;
		params.ir_intensity = 4;
	}

	cam.Open(params);

	cout << endl;
	if (!cam.IsOpened()) {
		cerr << "Error: Open camera failed" << endl;
		return 1;
	}
	cout << "Open device success" << endl << endl;

	float depth_scale = 0.001;

	int YELLOW_DISTANCE = (int) (_YELLOW_DISTANCE_IN_METERS * 1000);
	int RED_DISTANCE = (int)  (_RED_DISTANCE_IN_METERS * 1000);



	// ---------------------------------------------------------
	//  END:  Start MYNT-EYE-D
	// ---------------------------------------------------------


	// ---------------------------------------------------------
	//  BEGIN:  Start NN
	// ---------------------------------------------------------
	std::string modelPath = std::string(nn_weightfile);
	std::string configPath = std::string(nn_cfgfile);

	CDNeuralNet cdNet(modelPath, configPath);

	// Run the neural network in a different thread:
	std::thread worker(worker_thread, cdNet);

	// ---------------------------------------------------------
	//  END:  Start Darknet
	// ---------------------------------------------------------

	//Initialize new OpenCV window to output bounding boxes
	cv::namedWindow("MYNT-EYE-D", cv::WINDOW_NORMAL); 

	cv::Mat imD;
	int count=0;
	while (true) {

		cam.WaitForStream();

		auto image_color = cam.GetStreamData(ImageType::IMAGE_LEFT_COLOR);
		if (image_color.img) {
			imRGB = image_color.img->To(ImageFormat::COLOR_BGR)->ToMat();
		}

		auto image_depth = cam.GetStreamData(ImageType::IMAGE_DEPTH);
		if (image_depth.img) {
			imD = image_depth.img->To(ImageFormat::DEPTH_RAW)->ToMat();
		}


		for (int i=0; i<imD.rows; ++i) {
			for (int j=0; j<imD.cols; ++j) {
				// A little counter-intuitive, but:
				// The depth image has "shadows".  The Intel librealsense2 driver interprets
				// shadows as distance == 0.  But we will change that to distance=max, so that
				// everything else will ignore the shadows:
				if (imD.at<uint16_t>(i, j) < 20) {
					imD.at<uint16_t>(i, j) = 65535;
				}
			}
		}

		// White bounding box around the "object-too-close" detector:
		int depth_min = 200000;  // in integer depth_scale units
		int all_depth_min = 200000;  // in integer depth_scale units

		int nboxes_a;
		struct _bbox bboxes_a[25];

		// Copy the bounding boxes provided from the neural-network
		nboxes_a = cdNet.get_output_boxes(bboxes_a, 25);


		float half_w = _RGB_WIDTH / 2.0;
		float half_h = _RGB_HEIGHT / 2.0;


		//printf("nboxes: %d\n", nboxes_a);
		float x_mid,y_mid;

		for (int i=0; i<nboxes_a; ++i) {

			float x_center = bboxes_a[i].x * _RGB_WIDTH;
			float y_center = bboxes_a[i].y * _RGB_HEIGHT;
			float width = bboxes_a[i].w * _RGB_WIDTH;
			float height = bboxes_a[i].h * _RGB_HEIGHT;

			int x_min = (int) (x_center - bboxes_a[i].w * half_w);
			int x_max = (int) (x_min + width);
			int y_min = (int) (y_center - bboxes_a[i].h * half_h);
			int y_max = (int) (y_min + height);
			//printf("%d %d %d %d\n", x_min, x_max, y_min, y_max);
			// We won't scan close to the borders of the bounding boxes
			float scanFactor = 0.4;

			int scanLeft = (int) (x_center - bboxes_a[i].w * half_w * scanFactor);
			int scanRight = (int) (x_center + bboxes_a[i].w * half_w * scanFactor);

			int scanTop = (int) (y_center - bboxes_a[i].h * half_h * scanFactor);
			int scanBottom = (int) (y_center + bboxes_a[i].h * half_h * scanFactor);

			if (scanTop < 2) {
				scanTop = 2;
			}

			if (x_min < 0) {
				x_min = 0;
			}
			if (x_max >= _RGB_WIDTH) {
				x_max = _RGB_WIDTH - 1;
			}
			if (y_min < 0) {
				y_min = 0;
			}
			if (y_max >= _RGB_HEIGHT) {
				y_max = _RGB_HEIGHT - 1;
			}
			
		
			
			// Find the closest point within this box:
			depth_min = 65535;  // in integer depth_scale units
			uint16_t _d;
			for (int ii=scanTop; ii<scanBottom; ++ii) {
				for (int jj=scanLeft; jj<scanRight; ++jj) {

					_d = imD.at<uint16_t>(ii, jj);

					if (_d < depth_min) {
						depth_min = _d;
					}
				}
			}

			if (depth_min < all_depth_min) {
				all_depth_min = depth_min;
				x_mid = bboxes_a[i].x * _RGB_WIDTH;
			 	y_mid = bboxes_a[i].y * _RGB_HEIGHT;
				trajectory.push_back(cv::Point(x_mid,y_mid));
				if (trajectory.size()>150) 
					trajectory.pop_front();

			}

			cv::Rect r = cv::Rect(x_min, y_min, width, height);
			cv::rectangle(imRGB, r, cv::Scalar(0,0,200), 11);
			// printf("x : %.02f y : %.02f traj size : %lu\n",x_center,y_center,trajectory.size());
		}

		float closest = all_depth_min * depth_scale;
		char textBuffer1[255],textBuffer2[255];
		auto font = cv::FONT_HERSHEY_SIMPLEX;

		if (closest < _MAX_DISPLAY_DISTANCE_IN_METERS) {
			sprintf(textBuffer1, "Distance : %.02f m", closest);
			cv::putText(imRGB, textBuffer1, cv::Point(10,70), font, 1.5, cv::Scalar(255,255,255), 8);  // white text

		}
		
		sprintf(textBuffer2, "X : %.02f   Y : %.02f",x_mid,y_mid);
		cv::putText(imRGB, textBuffer2, cv::Point(10,140), font, 1.5, cv::Scalar(255,255,255), 8);  // white text
		for (int t=1;t<trajectory.size();t+=3){
			cv::line(imRGB,trajectory[t-1],trajectory[t],cv::Scalar(255,0,0),10);}
	

		cv::setWindowProperty("MYNT-EYE-D", cv::WND_PROP_AUTOSIZE, cv::WINDOW_NORMAL);
		cv::imshow("MYNT-EYE-D", imRGB);
		cv::waitKey(1);
		count++;
	}

	return 0;
}


