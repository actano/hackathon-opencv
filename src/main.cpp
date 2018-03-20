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

vector<vector<Point>> get_boxes(const vector<vector<Point>> &contours) {
    vector<vector<Point>> result;
    vector<Point> contour1;
    contour1.push_back(Point(350, 150));
    contour1.push_back(Point(350, 200));
    contour1.push_back(Point(200, 200));
    contour1.push_back(Point(200, 150));
    result.push_back(contour1);
    return result;
}

typedef struct _arrowData {
    vector<Point> lineSegment;
    vector<Point> contour;
} arrowData;

vector<arrowData> get_arrows(const vector<vector<Point>> &contours) {
    vector<arrowData> result;

    arrowData arrow1;
    arrow1.lineSegment.push_back(Point(350, 150));
    arrow1.lineSegment.push_back(Point(350, 150));

    arrow1.contour.push_back(Point(350, 150));

    result.push_back(arrow1);
    return result;
}

typedef struct _arrow {
    Point start;
    Point end;
} arrow;

int norm(Point p) {
    return (p.x)^2 + (p.y)^2;
}

int dist(Point p1, Point p2) {
    return norm(p1 - p2);
}

arrow approximate_arrow(const arrowData &data) {

    Point p1 = data.lineSegment[0];
    Point p2 = data.lineSegment[1];

    int near_p1 = 0;
    int near_p2 = 0;

    for(const Point& q: data.contour) {
        if (dist(p1, q) < dist(p2, q)) {
            near_p1++;
        }
        else {
            near_p2++;
        }
    }

    if(near_p2 > near_p1) {
        return {p1, p2};
    }
    return {p2, p1};
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
        if (poly.size() == 2) {
            arrow result = approximate_arrow({poly, contour});
            cout<<result.start<<endl;
            cout<<result.end<<endl;
        }
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
