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

typedef struct _BoundingBox {
    Point max = Point(INT_MIN, INT_MIN);;
    Point min = Point(INT_MAX, INT_MAX);
} BoundingBox;

BoundingBox getBoundingBox(const vector<Point> box) {
    BoundingBox boundingBox;
    for(Point point: box) {
        if(point.x < boundingBox.min.x) {
            boundingBox.min.x = point.x;
        }
        if(point.y < boundingBox.min.y) {
            boundingBox.min.y = point.y;
        }
        if(point.x > boundingBox.max.x) {
            boundingBox.max.x = point.x;
        }
        if(point.y > boundingBox.max.y) {
            boundingBox.max.y = point.y;
        }
    }

    return boundingBox;
}

bool lessThan(Point point1, Point point2) {
    return point1.x < point2.x && point1.y < point2.y;
}

bool greaterThan(Point point1, Point point2) {
    return point1.x > point2.x && point1.y > point2.y;
}

bool containsBox(const vector<Point>& box1, const vector<Point>& box2) {
    BoundingBox boundingBox1 = getBoundingBox(box1);
    BoundingBox boundingBox2 = getBoundingBox(box2);

    return lessThan(boundingBox1.min, boundingBox2.min) && greaterThan(boundingBox1.max, boundingBox2.max);
}

vector<vector<Point>> get_boxes(const vector<vector<Point>> &contours) {
    vector<vector<Point>> boxes;

    for(const vector<Point>& contour: contours) {
        vector<Point> poly;
        approxPolyDP(contour, poly, 50, true);
        if (poly.size() == 4) {
            boxes.push_back(poly);
        }
    }

    vector<vector<Point>> filteredBoxes;
    for(const vector<Point> box: boxes) {
        bool containsOtherBox = false;
        for(const vector<Point> otherBox: boxes) {
            if(containsBox(box, otherBox)) {
                containsOtherBox = true;
            }
        }
        if (!containsOtherBox) {
            filteredBoxes.push_back(box);
        }
    }

    return filteredBoxes;
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

void printContour(const vector<vector<Point>>& contours) {
    for(const vector<Point>& contour: contours) {
        cout<<"begin contour"<<endl;
        for(const Point& point: contour) {
            cout<<point<< endl;
        }
        cout<<"end contour"<<endl;
    }
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


    vector<vector<Point>> boxes = get_boxes(contours);

    printContour(boxes);

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
