#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "yolov8.h"
#include "byteTrack/BYTETracker.h"
#include <random>
#include "opencv2/opencv.hpp"
#include "yolov8-pose.hpp"
#include <chrono>

static vector<Scalar> colors;

void draw_bboxes(Mat& frame, const vector<byte_track::BYTETracker::STrackPtr> &output);
void format_tracker_input(Mat &frame, vector<Detection> &detections, vector<byte_track::Object> &tracker_objects);

const std::vector<std::vector<unsigned int>> KPS_COLORS = { {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255} };

const std::vector<std::vector<unsigned int>> SKELETON = { {16, 14},
                                                         {14, 12},
                                                         {17, 15},
                                                         {15, 13},
                                                         {12, 13},
                                                         {6, 12},
                                                         {7, 13},
                                                         {6, 7},
                                                         {6, 8},
                                                         {7, 9},
                                                         {8, 10},
                                                         {9, 11},
                                                         {2, 3},
                                                         {1, 2},
                                                         {1, 3},
                                                         {2, 4},
                                                         {3, 5},
                                                         {4, 6},
                                                         {5, 7} };

const std::vector<std::vector<unsigned int>> LIMB_COLORS = { {51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0} };

int main(int argc, char** argv)
{
    const string engine_path{ argv[1] };
    const string video_path{ argv[2] };
    assert(argc == 3);

    // Init model
    Yolov8 model(engine_path);

    // Init tracker
    byte_track::BYTETracker tracker(30, 30);

    // Store random colors
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(100, 255);
    for (int i = 0; i < 100; i++)
    {
        Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        colors.push_back(color);
    }

    // Open the video
    VideoCapture cap(video_path);

    // Create a VideoWriter object to save the processed video
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);    
    VideoWriter output_video("output_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(width, height));

    // pose添加
    auto yolov8_pose = new YOLOv8_pose(engine_path);
    yolov8_pose->make_pipe(true);
    std::vector<Detection> objs;
    cv::Size size = cv::Size{ 640, 640 };

    // Process the video
    while (1)
    {
        Mat frame,res;
        vector<Detection> detections;
        vector<byte_track::Object> tracks;
        cap >> frame;
        if (frame.empty()) break;

        /* pose添加 */
        objs.clear();
        yolov8_pose->copy_from_Mat(frame, size);
        yolov8_pose->infer();
        yolov8_pose->postprocess(objs, 0.25f, 0.35f, 100);
        yolov8_pose->draw_objects(frame, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
        //imshow("pose", res);
        /* pose添加 */

        //v9->detections, v8->objs
        //model.predict(frame, detections); 
        //std::cout << "/************************************/" <<  std::endl;
        format_tracker_input(frame, objs, tracks);
        const auto outputs = tracker.update(tracks);
        draw_bboxes(res, outputs);
        imshow("prediction", res);
        output_video.write(res);
        waitKey(1);
    }

    // Release resources
    destroyAllWindows();
    cap.release();
    output_video.release();

    return 0;
}

void format_tracker_input(Mat &frame, vector<Detection> &detections, vector<byte_track::Object> &tracker_objects)
{
    const float H = 640;
    const float W = 640;
    const float r_h = H / (float)frame.rows;
    const float r_w = W / (float)frame.cols;

    for (int i = 0; i < detections.size(); i++)
    {
        float x = detections[i].bbox.x;
        float y = detections[i].bbox.y;
        float width = detections[i].bbox.width;
        float height = detections[i].bbox.height;

        /* yolov8需开放这段后处理 */
        //if (r_h > r_w)
        //{
        //    x = x / r_w;
        //    y = (y - (H - r_w * frame.rows) / 2) / r_w;
        //    width = width / r_w;
        //    height = height / r_w;
        //}
        //else
        //{
        //    x = (x - (W - r_h * frame.cols) / 2) / r_h;
        //    y = y / r_h;
        //    width = width / r_h;
        //    height = height / r_h;
        //}
        /* yolov8需开放这段后处理 */

        byte_track::Rect<float> rect(x, y, width, height);

        byte_track::Object obj(rect, detections[i].class_id, detections[i].conf);

        tracker_objects.push_back(obj);
     }
}

void draw_bboxes(Mat& frame, const vector<byte_track::BYTETracker::STrackPtr> &output)
{
    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto trackId = detection->getTrackId();

        int x = detection->getRect().tlwh[0];
        int y = detection->getRect().tlwh[1];
        int width = detection->getRect().tlwh[2];
        int height = detection->getRect().tlwh[3];

        auto color_id = trackId % colors.size();
        rectangle(frame, Point(x, y), Point(x + width, y + height), colors[color_id], 3);

        // Detection box text
        string classString = to_string(trackId);
        Size textSize = getTextSize(classString, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect textBox(x, y - 40, textSize.width + 10, textSize.height + 20);
        rectangle(frame, textBox, colors[color_id], FILLED);
        putText(frame, classString, Point(x + 5, y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}
