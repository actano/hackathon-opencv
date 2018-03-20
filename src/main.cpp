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

//begin arrow
//begin contour
//[390, 256]
//[391, 255]
//[393, 255]
//[394, 256]
//[396, 256]
//[397, 257]
//[398, 257]
//[400, 259]
//[401, 259]
//[404, 262]
//[405, 262]
//[406, 263]
//[407, 263]
//[408, 264]
//[409, 264]
//[410, 265]
//[412, 265]
//[413, 266]
//[414, 266]
//[415, 267]
//[417, 267]
//[419, 269]
//[420, 269]
//[422, 271]
//[423, 271]
//[429, 277]
//[430, 277]
//[433, 280]
//[434, 280]
//[435, 281]
//[436, 281]
//[440, 285]
//[441, 285]
//[444, 288]
//[445, 288]
//[448, 291]
//[449, 291]
//[450, 292]
//[451, 292]
//[456, 297]
//[457, 297]
//[459, 299]
//[460, 299]
//[461, 300]
//[462, 300]
//[463, 301]
//[465, 301]
//[468, 304]
//[469, 304]
//[472, 307]
//[473, 307]
//[477, 311]
//[478, 311]
//[480, 313]
//[481, 313]
//[482, 314]
//[482, 315]
//[483, 315]
//[486, 318]
//[487, 318]
//[489, 320]
//[490, 320]
//[492, 322]
//[493, 322]
//[496, 325]
//[497, 325]
//[503, 331]
//[504, 331]
//[506, 333]
//[507, 333]
//[509, 335]
//[510, 335]
//[511, 336]
//[512, 336]
//[513, 337]
//[514, 337]
//[518, 341]
//[519, 341]
//[522, 344]
//[523, 344]
//[524, 345]
//[525, 345]
//[526, 346]
//[530, 346]
//[532, 344]
//[533, 344]
//[534, 343]
//[535, 343]
//[536, 342]
//[538, 342]
//[539, 343]
//[539, 345]
//[541, 347]
//[541, 348]
//[542, 349]
//[542, 351]
//[543, 352]
//[543, 354]
//[544, 355]
//[544, 357]
//[545, 358]
//[545, 359]
//[546, 360]
//[546, 365]
//[547, 366]
//[547, 378]
//[546, 379]
//[545, 379]
//[544, 380]
//[543, 380]
//[541, 382]
//[536, 382]
//[535, 381]
//[532, 381]
//[531, 380]
//[530, 380]
//[529, 379]
//[528, 379]
//[527, 378]
//[523, 378]
//[522, 379]
//[521, 379]
//[520, 378]
//[519, 378]
//[518, 377]
//[517, 377]
//[516, 376]
//[513, 376]
//[510, 373]
//[510, 369]
//[514, 365]
//[515, 365]
//[518, 362]
//[519, 362]
//[520, 361]
//[522, 361]
//[522, 358]
//[521, 357]
//[520, 357]
//[518, 355]
//[517, 355]
//[516, 354]
//[515, 354]
//[513, 352]
//[512, 352]
//[511, 351]
//[510, 351]
//[508, 349]
//[507, 349]
//[505, 347]
//[504, 347]
//[500, 343]
//[499, 343]
//[498, 342]
//[498, 341]
//[497, 341]
//[496, 340]
//[495, 340]
//[491, 336]
//[490, 336]
//[486, 332]
//[485, 332]
//[481, 328]
//[480, 328]
//[478, 326]
//[477, 326]
//[473, 322]
//[472, 322]
//[468, 318]
//[467, 318]
//[463, 314]
//[462, 314]
//[461, 313]
//[460, 313]
//[459, 312]
//[458, 312]
//[456, 310]
//[455, 310]
//[454, 309]
//[453, 309]
//[448, 304]
//[447, 304]
//[445, 302]
//[444, 302]
//[440, 298]
//[439, 298]
//[437, 296]
//[436, 296]
//[435, 295]
//[434, 295]
//[424, 285]
//[423, 285]
//[420, 282]
//[419, 282]
//[417, 280]
//[416, 280]
//[414, 278]
//[413, 278]
//[410, 275]
//[409, 275]
//[408, 274]
//[407, 274]
//[405, 272]
//[404, 272]
//[402, 270]
//[401, 270]
//[400, 269]
//[399, 269]
//[398, 268]
//[396, 268]
//[394, 266]
//[393, 266]
//[392, 265]
//[391, 265]
//[390, 264]
//[389, 264]
//[388, 263]
//[386, 263]
//[385, 262]
//[384, 262]
//[383, 261]
//[383, 260]
//[385, 258]
//[387, 258]
//[389, 256]
//end contour
//begin line segment
//[383, 260]
//[547, 378]
//end line segmentv

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

arrow approximate_arrow(const arrowData &arrow) {
    return {
        Point(350, 150),
        Point(550, 250),
    };
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
            cout<<"begin arrow"<<endl;
            cout<<"begin contour"<<endl;
            for(const Point& point: contour) {
                cout<<point<< endl;
            }
            cout<<"end contour"<<endl;
            cout<<"begin line segment"<<endl;
            for(const Point& point: poly) {
                cout<<point<< endl;
            }
            cout<<"end line segment"<<endl;
            cout<<"end arrow"<<endl;
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
