#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace cv;
using namespace std;

RNG rng(12345);
vector<vector<Point>> recognise_shape(InputArray in, OutputArray drawing);
void draw_contour(const vector<vector<Point>> &contours, Size size, OutputArray out);

void preprocess(InputArray in, OutputArray out) {
    const int thresh = 150;
    cvtColor(in, out, COLOR_BGR2GRAY);
    blur(out, out, Size(3,3));
    threshold(out, out, thresh, 255, THRESH_BINARY);
}

int main( int, char** argv )
{
    Mat src = imread(argv[1]);
    if (src.empty())
    {
        cerr << "No image supplied ..." << endl;
        return -1;
    }

    Mat preprocessed;
    preprocess(src, preprocessed);

    const char* source_window = "Source";
    namedWindow( source_window, WINDOW_AUTOSIZE );
    imshow( source_window, preprocessed );

    Mat contour_drawing;
    vector<vector<Point>> contours = recognise_shape(preprocessed, contour_drawing);
    namedWindow( "Contour", WINDOW_AUTOSIZE );
    imshow( "Contour", contour_drawing );

    vector<vector<Point>> polygons;
    for(const vector<Point>& contour: contours) {
        vector<Point> poly;
        approxPolyDP(contour, poly, 50, true);
        polygons.push_back(poly);
    }

    Mat poly_drawing;
    draw_contour(polygons, contour_drawing.size(), poly_drawing);

    namedWindow( "Poly", WINDOW_AUTOSIZE );
    imshow( "Poly", poly_drawing );

    waitKey(0);
    return(0);
}

void draw_contour(const vector<vector<Point>> &contours, Size size, OutputArray out) {
    Mat drawing = Mat::zeros( size, CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, (int)i, color, 2, 8);
    }
    out.assign(drawing);
}

vector<vector<Point>> recognise_shape(InputArray in, OutputArray out)
{
    vector<Vec4i> hierarchy;
    vector<vector<Point>> contours;
    findContours(in.getMat(), contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    draw_contour(contours, in.size(), out);
    return contours;
}
