#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace cv;
using namespace std;


typedef struct _arrow {
    Point start;
    Point end;
} arrow;

typedef struct _link {
    vector<Point> boxStart;
    vector<Point> boxEnd;
    arrow linkBetween;
} link;

template <typename T>
std::ostream& operator<<(std::ostream& os, const vector<T> &t);

std::ostream& operator<<(std::ostream& os, const link &l);

std::ostream& operator<<(std::ostream& os, const arrow &a);

RNG rng(12345);
vector<vector<Point>> recognise_shape(InputArray in, OutputArray drawing);
void draw_contour(const vector<vector<Point>> &contours, Size size, OutputArray out);
void draw_arrow_heads(const vector<arrow> &arrows, Size size, InputOutputArray canvas);
void drawLinks(const vector<link> &links, InputOutputArray canvas);

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

    for(const vector<Point>& contour: contours) {
        vector<Point> poly;
        approxPolyDP(contour, poly, 50, true);
        if (poly.size() == 2) {
            arrowData arrowD;
            arrowD.lineSegment = poly;
            arrowD.contour = contour;
            result.push_back(arrowD);
        }
    }

    return result;
}

int norm(Point p) {
    return p.x * p.x + p.y * p.y;
}

int dist(Point p1, Point p2) {
    return norm(p1 - p2);
}

double area(InputArray points) {
    vector<Point> tmp;
    convexHull(points, tmp);
    return contourArea(tmp);
}

arrow approximate_arrow(const arrowData &data) {

    Point p1 = data.lineSegment[0];
    Point p2 = data.lineSegment[1];

    vector<Point> poly;
    approxPolyDP(data.contour, poly, 10, true);

    vector<Point> near_p1;
    vector<Point> near_p2;

    for(const Point& q: poly) {
        if (dist(p1, q) < dist(p2, q)) {
            near_p1.push_back(q);
        }
        else {
            near_p2.push_back(q);
        }
    }

    if(near_p2.size() > near_p1.size()) {
        return {p1, p2};
    }
    return {p2, p1};
}

void printC(const vector<Point> &contour) {
    cout<<"begin contour"<<endl;
    for(const Point& point: contour) {
        cout<<point<< endl;
    }
    cout<<"end contour"<<endl;
}

void printContour(const vector<vector<Point>>& contours) {
    for(const vector<Point>& contour: contours) {
        printC(contour);
    }
}

void printArrows(const vector<arrowData>& arrows) {
    for(const arrowData &arrow: arrows) {
        cout<<"begin arrow"<<endl;
        printC(arrow.lineSegment);
        printC(arrow.contour);
        cout<<"end arrow"<<endl;
    }
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const vector<T> &t)
    {
        for(const T &e: t) {
            os << e << endl;
        }
        os << endl;
        return os;
    }

std::ostream& operator<<(std::ostream& os, const link &l)
    {
        os << "Link:" << endl << "StartBox: " << endl << l.boxStart << endl;
        os <<  "EndBox:" << endl << l.boxEnd << endl;
        os << "Arrow: " << endl << l.linkBetween << endl;
        return os;
    }

std::ostream& operator<<(std::ostream& os, const arrow &a)
    {
        os <<  "Start:" << endl << a.start << endl;
        os << "End: " << endl << a.end << endl;
        return os;
    }


vector<Point> findClosestBox(const vector<vector<Point>> &boxes, Point point) {
    vector<Point> closestBox = boxes.front();
    int minDist = INT_MAX;
    for(const vector<Point> &box: boxes) {
        cout << minDist << " ";
        for(const Point &boxPoint: box) {
            if (dist(boxPoint, point) < minDist) {
                minDist = dist(boxPoint, point);
                closestBox = box;
            }
            cout << minDist << " ";
        }
        cout << endl;
    }

    return closestBox;
}

vector<link> getLinks(const vector<vector<Point>> &boxes, const vector<arrow> &arrows) {
    vector<link> links;
    cout << "Boxes " << boxes << endl;
    cout << "Arrows " << arrows << endl;
    for(const arrow &nextArrow: arrows) {
        link newLink;
        newLink.boxStart = findClosestBox(boxes, nextArrow.start);
        newLink.boxEnd = findClosestBox(boxes, nextArrow.end);
        newLink.linkBetween = nextArrow;
        links.push_back(newLink);
    }
    return links;
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

    vector<vector<Point>> boxes = get_boxes(contours);
    vector<arrowData> arrowDataVector = get_arrows(contours);
    vector<arrow> arrows;

    for(const arrowData& elm: arrowDataVector) {
        arrows.push_back(approximate_arrow(elm));
    }

    vector<link> links = getLinks(boxes, arrows);
    drawLinks(links, contour_drawing);

    namedWindow( "Contour", WINDOW_AUTOSIZE );
    imshow( "Contour", contour_drawing );

  //  printContour(boxes);
  //  printArrows(arrowDataVector);
    cout << links << endl;

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

void draw_arrow_heads(const vector<arrow> &arrows, Size size, InputOutputArray canvas) {
   for (const arrow& arrow: arrows)
       {
           Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
           circle(canvas, arrow.end, 10, color, 3, 0, 0);
           circle(canvas, arrow.start, 5, color, 3, 0, 0);
       }
}

void drawLinks(const vector<link> &links, InputOutputArray canvas) {
   for(const link &l: links) {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        vector<vector<Point>> contours;
        contours.push_back(l.boxStart);

        drawContours(canvas, contours, -1, color, 2, 8);

       contours.pop_back();
       contours.push_back(l.boxEnd);

       drawContours(canvas, contours, -1, color, 8, 8);

        circle(canvas, l.linkBetween.end, 10, color, 8, 0, 0);
        circle(canvas, l.linkBetween.start, 5, color, 3, 0, 0);
   }
}

vector<vector<Point>> recognise_shape(InputArray in, OutputArray out)
{
    vector<Vec4i> hierarchy;
    vector<vector<Point>> contours;
    findContours(in.getMat(), contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    draw_contour(contours, in.size(), out);
    return contours;
}
