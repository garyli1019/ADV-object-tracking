#include <stdio.h>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sstream>
#include <string>
#include <iostream>
#include <curl/curl.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>

using namespace cv;
using namespace std;

//initial threshold
int H_MAX=255;
int H_MIN=0;
int S_MAX=255;
int S_MIN=0;
int V_MIN=0;
int V_MAX=255;
// threshold for blue
int H_MIN_b = 41;
int H_MAX_b = 109;
int S_MIN_b = 100;
int S_MAX_b = 209;
int V_MIN_b = 84;
int V_MAX_b = 224;

// threshold for red
int H_MIN_r = 0;
int H_MAX_r = 20;
int S_MIN_r = 180;
int S_MAX_r = 250;
int V_MIN_r = 120;
int V_MAX_r = 220;

//Global switch
int findobj=1;   //enable to track object
int qr_able=0;   //enable to find qr code
int lift_on=0;   //enable to the lift
int time_cout=0;   //time counter for lift
int track_blue=1;
int track_red=0;
//IP address
string im_addr="http://192.168.0.100:8091/?action=snapshot";
string ser_addr="192.168.0.101";
//default capture width and height
const int FRAME_WIDTH = 320;
const int FRAME_HEIGHT = 240;

//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS=10;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;
//names that will appear at the top of each window
const String windowName = "Original Image";
const String windowName1 = "HSV Image";
const String windowName2 = "Thresholded Image";


//Variables for QR code detection
const int CV_QR_NORTH = 0;
const int CV_QR_EAST = 1;
const int CV_QR_SOUTH = 2;
const int CV_QR_WEST = 3;


//*****************************************************************************
//function initial for QR code detection

float cv_distance(Point2f P, Point2f Q);					// Get Distance between two points
float cv_lineEquation(Point2f L, Point2f M, Point2f J);		// Perpendicular Distance of a Point J from line formed by Points L and M; Solution to equation of the line Val = ax+by+c
float cv_lineSlope(Point2f L, Point2f M, int& alignement);	// Slope of a line by two Points L and M on it; Slope of line, S = (x1 -x2) / (y1- y2)
void cv_getVertices(vector<vector<Point> > contours, int c_id,float slope, vector<Point2f>& X);
void cv_updateCorner(Point2f P, Point2f ref ,float& baseline,  Point2f& corner);
void cv_updateCornerOr(int orientation, vector<Point2f> IN, vector<Point2f> &OUT);
bool getIntersectionPoint(Point2f a1, Point2f a2, Point2f b1, Point2f b2, Point2f& intersection);
float cross(Point2f v1,Point2f v2);


// *****************************************************************************
// Function for CRUL
struct memoryStruct {
    char *memory;
    size_t size;
};

static void* CURL_realloc(void *ptr, size_t size)
{
    /* There might be a realloc() out there that doesn't like reallocing
     NULL pointers, so we take care of it here */
    if(ptr)
        return realloc(ptr, size);
    else
        return malloc(size);
}

size_t WriteMemoryCallback
(void *ptr, size_t size, size_t nmemb, void *data)
{
    size_t realsize = size * nmemb;
    struct memoryStruct *mem = (struct memoryStruct *)data;
    
    mem->memory = (char *)
    CURL_realloc(mem->memory, mem->size + realsize + 1);
    if (mem->memory) {
        memcpy(&(mem->memory[mem->size]), ptr, realsize);
        mem->size += realsize;
        mem->memory[mem->size] = 0;
    }
    return realsize;
}

Mat Get_Image(const char* url)
{
    CURL *curl;       // CURL objects
    CURLcode res;
    cv::Mat imgTmp; 	// image object
    memoryStruct buffer; // memory buffer
    
    curl = curl_easy_init(); // init CURL library object/structure
    
    
    
    if(curl) {
        
        // set up the write to memory buffer
        // (buffer starts off empty)
        
        buffer.memory = NULL;
        buffer.size = 0;
        
        // (N.B. check this URL still works in browser in case image has moved)
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 0); // disable printing info
        // tell libcurl where to write the image (to a dynamic memory buffer)
        curl_easy_setopt(curl,CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
        curl_easy_setopt(curl,CURLOPT_WRITEDATA, (void *) &buffer);
        
        // get the image from the specified URL
        
        res = curl_easy_perform(curl);
        
        // decode memory buffer using OpenCV
        
        imgTmp = cv::imdecode(cv::Mat(1, buffer.size, CV_8UC1, buffer.memory), CV_LOAD_IMAGE_UNCHANGED);
        
        
        
        
        
        // always cleanup
        
        curl_easy_cleanup(curl);
        free(buffer.memory);
        
        
    };
    return imgTmp;
}

// *****************************************************************************
// Function for Object Tracking


String intToString(int number){
    
    
    std::stringstream ss;
    ss << number;
    return ss.str();
}

void drawObject(int x, int y,Mat &frame){
    
    //use some of the openCV drawing functions to draw crosshairs
    //on your tracked image!
    
    //UPDATE:JUNE 18TH, 2013
    //added 'if' and 'else' statements to prevent
    //memory errors from writing off the screen (ie. (-25,-25) is not within the window!)
    
    circle(frame,Point(x,y),20,Scalar(0,255,0),2);
    if(y-25>0)
        line(frame,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(x,0),Scalar(0,255,0),2);
    if(y+25<FRAME_HEIGHT)
        line(frame,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(x,FRAME_HEIGHT),Scalar(0,255,0),2);
    if(x-25>0)
        line(frame,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(0,y),Scalar(0,255,0),2);
    if(x+25<FRAME_WIDTH)
        line(frame,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(FRAME_WIDTH,y),Scalar(0,255,0),2);
    
    putText(frame,intToString(x)+","+intToString(y),Point(x,y+30),1,1,Scalar(0,255,0),2);
    
}
void morphOps(Mat &thresh){
    
    //create structuring element that will be used to "dilate" and "erode" image.
    //the element chosen here is a 3px by 3px rectangle
    
    Mat erodeElement = getStructuringElement( MORPH_RECT,Size(3,3));
    //dilate with larger element so make sure object is nicely visible
    Mat dilateElement = getStructuringElement( MORPH_RECT,Size(8,8));
    
    erode(thresh,thresh,erodeElement);
    erode(thresh,thresh,erodeElement);
    
    
    dilate(thresh,thresh,dilateElement);
    dilate(thresh,thresh,dilateElement);
    
    
    
}
void trackFilteredObject(int &x, int &y,int area, Mat threshold, Mat &cameraFeed){
    
    Mat temp;
    threshold.copyTo(temp);
    //these two vectors needed for output of findContours
    std::vector< std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;
    //find contours of filtered image using openCV findContours function
    findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );
    //use moments method to find our filtered object
    double refArea = 0;
    bool objectFound = false;
    if (hierarchy.size() > 0) {
        int numObjects = hierarchy.size();
        //if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
        if(numObjects<MAX_NUM_OBJECTS){
            for (int index = 0; index >= 0; index = hierarchy[index][0]) {
                
                Moments moment = moments((cv::Mat)contours[index]);
                double area = moment.m00;
                
                //if the area is less than 20 px by 20px then it is probably just noise
                //if the area is the same as the 3/2 of the image size, probably just a bad filter
                //we only want the object with the largest area so we safe a reference area each
                //iteration and compare it to the area in the next iteration.
                if(area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea){
                    x = moment.m10/area;
                    y = moment.m01/area;
                    objectFound = true;
                    refArea = area;
                }else objectFound = false;
                
                
            }
            //let user know you found an object
            if(objectFound ==true){
                putText(cameraFeed,"Tracking Object",Point(0,50),2,1,Scalar(0,255,0),2);
                //draw object location on screen
                drawObject(x,y,cameraFeed);}
            
        }else putText(cameraFeed,"TOO MUCH NOISE! ADJUST FILTER",Point(0,50),1,2,Scalar(0,0,255),2);
    }
}




// *****************************************************************************

int main(int argc, char* argv[])
{
    
    int primarySocket;
    struct sockaddr_in serv_addr;
    char buffer[256];
    const int serverPort = 3000;
    
    primarySocket = socket(AF_INET , SOCK_STREAM , 0);
    
    printf("Initializing connection... \n");
    
    if (primarySocket < 0) {
        perror("Can't create socket");
        return 1;
    }
    
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_addr.s_addr = inet_addr("192.168.0.100");
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(serverPort);
    
    if (connect(primarySocket, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0){
        perror("Can't connect");
        return 1;
    }
    
    printf("Connected\n");
    
    Mat HSV;
    //matrix storage for binary threshold image
    Mat thresholdimg;
    
    while(findobj||qr_able){
    while(findobj){
        
        int n = 0;
        bzero(buffer,256);
        int x=0, y=0,area=0;
        
        
        //convert frame from RGB to HSV colorspace
        Mat I=Get_Image("http://192.168.0.100:8091/?action=snapshot");
        
        cvtColor(I,HSV,COLOR_BGR2HSV);
        // set up threshold for blue or red
        if(track_blue){
            H_MAX=H_MAX_b;
            H_MIN=H_MIN_b;
            S_MAX=S_MAX_b;
            S_MIN=S_MIN_b;
            V_MIN=V_MIN_b;
            V_MAX=V_MAX_b;
            
        }
        else if(track_red){
            H_MAX=H_MAX_r;
            H_MIN=H_MIN_r;
            S_MAX=S_MAX_r;
            S_MIN=S_MIN_r;
            V_MIN=V_MIN_r;
            V_MAX=V_MAX_r;
            
        }
        //threshold matrix
        inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),thresholdimg);
        
        //perform morphological operations on thresholded image
        morphOps(thresholdimg);
        
        //track object
        trackFilteredObject(x,y,area,thresholdimg,I);
        
        //control the vehicle
        //blue object detect
    if(y<220&&track_blue){
        //stop
        if (x==0 && y==0){
            buffer[0] = 'h';
            buffer[1] = '\n';
            buffer[2] = '\0';
            if(send(primarySocket, buffer, strlen(buffer), 0) < 0) {
                perror("Can't send");
                
            }
        }
        //left
        else if (x<=120){
            buffer[0] = 'a';
            buffer[1] = '\n';
            buffer[2] = '\0';
            if(send(primarySocket, buffer, strlen(buffer), 0) < 0) {
                perror("Can't send");
                //return 1;
            }
            
        }
        //right
        else if(x>=240){
            buffer[0] = 'd';
            buffer[1] = '\n';
            buffer[2] = '\0';
            if(send(primarySocket, buffer, strlen(buffer), 0) < 0) {
                perror("Can't send");
                //return 1;
            }
        }
        //forward
        else{
            buffer[0] = 'w';
            buffer[1] = '\n';
            buffer[2] = '\0';
            if(send(primarySocket, buffer, strlen(buffer), 0) < 0) {
                perror("Can't send");
                //return 1;
            }
        }
    }
    //when close enough to object
    else if(y>=220&&track_blue) {
        buffer[0] = 'h';
        buffer[1] = '\n';
        buffer[2] = '\0';
        lift_on=1;
        track_blue=0;
        if(send(primarySocket, buffer, strlen(buffer), 0) < 0) {
            perror("Can't send");
            //return 1;
        }
    }
    if(lift_on && time_cout<40){
            buffer[0] = 'q';
            buffer[1] = '\n';
            buffer[2] = '\0';
            if(send(primarySocket, buffer, strlen(buffer), 0) < 0) {
                perror("Can't send");
                //return 1;
            }
            time_cout++;
    }
    else if(lift_on&& time_cout>=40){
            time_cout=0;
            lift_on=0;
            track_red=1;
            buffer[0] = ' ';
            buffer[1] = '\n';
            buffer[2] = '\0';
            if(send(primarySocket, buffer, strlen(buffer), 0) < 0) {
                perror("Can't send");
                //return 1;
            }
    }
        //when red object wasn't on screen
        if(track_red && x==0 && y==0){
            buffer[0] = 'd';
            buffer[1] = '\n';
            buffer[2] = '\0';
            if(send(primarySocket, buffer, strlen(buffer), 0) < 0) {
                perror("Can't send");
                //return 1;
            }
        }
        //found red object and far away from it
        else if(track_red && y<180){
            //left
            if (x<=120){
                buffer[0] = 'a';
                buffer[1] = '\n';
                buffer[2] = '\0';
                if(send(primarySocket, buffer, strlen(buffer), 0) < 0) {
                    perror("Can't send");
                    //return 1;
                }
                
            }
            //right
            else if(x>=240){
                buffer[0] = 'd';
                buffer[1] = '\n';
                buffer[2] = '\0';
                if(send(primarySocket, buffer, strlen(buffer), 0) < 0) {
                    perror("Can't send");
                    //return 1;
                }
            }
            //forward
            else{
                buffer[0] = 'w';
                buffer[1] = '\n';
                buffer[2] = '\0';
                if(send(primarySocket, buffer, strlen(buffer), 0) < 0) {
                    perror("Can't send");
                    //return 1;
                }
            }

        }
        //stop   FINAL DESTINATION OF MOVING!!!!!!!!!!!!!!!!!!!!!
        else if (track_red&&y>=180){
            buffer[0] = 'h';
            buffer[1] = '\n';
            buffer[2] = '\0';
            //set both switch = 0
            findobj=0;
            qr_able=1;
            cout << "Object Tracking Done !"<<endl;
            if(send(primarySocket, buffer, strlen(buffer), 0) < 0) {
                perror("Can't send");
                //return 1;
        }
        }
        bzero(buffer,256);
        n = recv(primarySocket, buffer, 255, 0);
        if(n < 0) {
            perror("Can't read");
            return 1;
        }
        else if(n == 0) {
            perror("Can't communicate with server");
            return 1;
        }
        printf("Server: %s\n\n", buffer);
        
        //display image
        imshow(windowName,I);
        //namedWindow("threshold", CV_WINDOW_AUTOSIZE);
        imshow(windowName2,thresholdimg);
        
        
        
        //delay 50ms so that screen can refresh.
        waitKey(50);
    }
    //QR code detection
    while(qr_able){
        
        Mat image=Get_Image("http://192.168.0.100:8091/?action=snapshot");
        
        //code from QR detection
        
        // Creation of Intermediate 'Image' Objects required later
        Mat gray(image.size(), CV_MAKETYPE(image.depth(), 1));			// To hold Grayscale Image
        Mat edges(image.size(), CV_MAKETYPE(image.depth(), 1));			// To hold Grayscale Image
        Mat traces(image.size(), CV_8UC3);								// For Debug Visuals
        Mat qr,qr_raw,qr_gray,qr_thres;
        
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        vector<Point> pointsseq;    //used to save the approximated sides of each contour
        
        int mark,A,B,C,top,right,bottom,median1,median2,outlier;
        float AB,BC,CA, dist,slope, areat,arear,areab, large, padding;
        
        int align,orientation;
        
        int DBG=1;						// Debug Flag
        
        int key = 0;
        while(key != 'q')				// While loop to query for Image Input frame
        {
            
            traces = Scalar(0,0,0);
            qr_raw = Mat::zeros(100, 100, CV_8UC3 );
            qr = Mat::zeros(100, 100, CV_8UC3 );
            qr_gray = Mat::zeros(100, 100, CV_8UC1);
            qr_thres = Mat::zeros(100, 100, CV_8UC1);
            
            image=Get_Image("http://192.168.0.100:8091/?action=snapshot");
            
            cvtColor(image,gray,CV_RGB2GRAY);		// Convert Image captured from Image Input to GrayScale
            Canny(gray, edges, 100 , 200, 3);		// Apply Canny edge detection on the gray image
            
            
            findContours( edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE); // Find contours with hierarchy
            
            mark = 0;								// Reset all detected marker count for this frame
            
            // Get Moments for all Contours and the mass centers
            vector<Moments> mu(contours.size());
            vector<Point2f> mc(contours.size());
            
            for( int i = 0; i < contours.size(); i++ )
            {	mu[i] = moments( contours[i], false );
                mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
            }
            
            
            // Start processing the contour data
            
            // Find Three repeatedly enclosed contours A,B,C
            // NOTE: 1. Contour enclosing other contours is assumed to be the three Alignment markings of the QR code.
            // 2. Alternately, the Ratio of areas of the "concentric" squares can also be used for identifying base Alignment markers.
            // The below demonstrates the first method
            
            for( int i = 0; i < contours.size(); i++ )
            {
                //Find the approximated polygon of the contour we are examining
                approxPolyDP(contours[i], pointsseq, arcLength(contours[i], true)*0.02, true);
                if (pointsseq.size() == 4)      // only quadrilaterals contours are examined
                {
                    int k=i;
                    int c=0;
                    
                    while(hierarchy[k][2] != -1)
                    {
                        k = hierarchy[k][2] ;
                        c = c+1;
                    }
                    if(hierarchy[k][2] != -1)
                        c = c+1;
                    
                    if (c >= 5)
                    {
                        if (mark == 0)		A = i;
                        else if  (mark == 1)	B = i;		// i.e., A is already found, assign current contour to B
                        else if  (mark == 2)	C = i;		// i.e., A and B are already found, assign current contour to C
                        mark = mark + 1 ;
                    }
                }
            }
            
            
            if (mark >= 3)		// Ensure we have (atleast 3; namely A,B,C) 'Alignment Markers' discovered
            {
                // We have found the 3 markers for the QR code; Now we need to determine which of them are 'top', 'right' and 'bottom' markers
                
                // Determining the 'top' marker
                // Vertex of the triangle NOT involved in the longest side is the 'outlier'
                
                AB = cv_distance(mc[A],mc[B]);
                BC = cv_distance(mc[B],mc[C]);
                CA = cv_distance(mc[C],mc[A]);
                
                if ( AB > BC && AB > CA )
                {
                    outlier = C; median1=A; median2=B;
                }
                else if ( CA > AB && CA > BC )
                {
                    outlier = B; median1=A; median2=C;
                }
                else if ( BC > AB && BC > CA )
                {
                    outlier = A;  median1=B; median2=C;
                }
                
                top = outlier;							// The obvious choice
                
                dist = cv_lineEquation(mc[median1], mc[median2], mc[outlier]);	// Get the Perpendicular distance of the outlier from the longest side
                slope = cv_lineSlope(mc[median1], mc[median2],align);		// Also calculate the slope of the longest side
                
                // Now that we have the orientation of the line formed median1 & median2 and we also have the position of the outlier w.r.t. the line
                // Determine the 'right' and 'bottom' markers
                
                if (align == 0)
                {
                    bottom = median1;
                    right = median2;
                }
                else if (slope < 0 && dist < 0 )		// Orientation - North
                {
                    bottom = median1;
                    right = median2;
                    orientation = CV_QR_NORTH;
                }
                else if (slope > 0 && dist < 0 )		// Orientation - East
                {
                    right = median1;
                    bottom = median2;
                    orientation = CV_QR_EAST;
                }
                else if (slope < 0 && dist > 0 )		// Orientation - South
                {
                    right = median1;
                    bottom = median2;
                    orientation = CV_QR_SOUTH;
                }
                
                else if (slope > 0 && dist > 0 )		// Orientation - West
                {
                    bottom = median1;
                    right = median2;
                    orientation = CV_QR_WEST;
                }
                
                
                // To ensure any unintended values do not sneak up when QR code is not present
                float area_top,area_right, area_bottom;
                
                if( top < contours.size() && right < contours.size() && bottom < contours.size() && contourArea(contours[top]) > 10 && contourArea(contours[right]) > 10 && contourArea(contours[bottom]) > 10 )
                {
                    
                    vector<Point2f> L,M,O, tempL,tempM,tempO;
                    Point2f N;
                    
                    vector<Point2f> src,dst;		// src - Source Points basically the 4 end co-ordinates of the overlay image
                    // dst - Destination Points to transform overlay image
                    
                    Mat warp_matrix;
                    
                    cv_getVertices(contours,top,slope,tempL);
                    cv_getVertices(contours,right,slope,tempM);
                    cv_getVertices(contours,bottom,slope,tempO);
                    
                    cv_updateCornerOr(orientation, tempL, L); 			// Re-arrange marker corners w.r.t orientation of the QR code
                    cv_updateCornerOr(orientation, tempM, M); 			// Re-arrange marker corners w.r.t orientation of the QR code
                    cv_updateCornerOr(orientation, tempO, O); 			// Re-arrange marker corners w.r.t orientation of the QR code
                    
                    int iflag = getIntersectionPoint(M[1],M[2],O[3],O[2],N);
                    
                    
                    src.push_back(L[0]);
                    src.push_back(M[1]);
                    src.push_back(N);
                    src.push_back(O[3]);
                    
                    dst.push_back(Point2f(0,0));
                    dst.push_back(Point2f(qr.cols,0));
                    dst.push_back(Point2f(qr.cols, qr.rows));
                    dst.push_back(Point2f(0, qr.rows));
                    
                    if (src.size() == 4 && dst.size() == 4 )			// Failsafe for WarpMatrix Calculation to have only 4 Points with src and dst
                    {
                        warp_matrix = getPerspectiveTransform(src, dst);
                        warpPerspective(image, qr_raw, warp_matrix, Size(qr.cols, qr.rows));
                        copyMakeBorder( qr_raw, qr, 10, 10, 10, 10,BORDER_CONSTANT, Scalar(255,255,255) );
                        
                        cvtColor(qr,qr_gray,CV_RGB2GRAY);
                        threshold(qr_gray, qr_thres, 127, 255, CV_THRESH_BINARY);
                        
                        //threshold(qr_gray, qr_thres, 0, 255, CV_THRESH_OTSU);
                        //for( int d=0 ; d < 4 ; d++){	src.pop_back(); dst.pop_back(); }
                    }
                    
                    //Draw contours on the image
                    drawContours( image, contours, top , Scalar(255,200,0), 2, 8, hierarchy, 0 );
                    drawContours( image, contours, right , Scalar(0,0,255), 2, 8, hierarchy, 0 );
                    drawContours( image, contours, bottom , Scalar(255,0,100), 2, 8, hierarchy, 0 );
                    
                    // Insert Debug instructions here
                    if(DBG==1)
                    {
                        // Debug Prints
                        // Visualizations for ease of understanding
                        if (slope > 5)
                            circle( traces, Point(10,20) , 5 ,  Scalar(0,0,255), -1, 8, 0 );
                        else if (slope < -5)
                            circle( traces, Point(10,20) , 5 ,  Scalar(255,255,255), -1, 8, 0 );
                        
                        // Draw contours on Trace image for analysis
                        drawContours( traces, contours, top , Scalar(255,0,100), 1, 8, hierarchy, 0 );
                        drawContours( traces, contours, right , Scalar(255,0,100), 1, 8, hierarchy, 0 );
                        drawContours( traces, contours, bottom , Scalar(255,0,100), 1, 8, hierarchy, 0 );
                        
                        // Draw points (4 corners) on Trace image for each Identification marker
                        circle( traces, L[0], 2,  Scalar(255,255,0), -1, 8, 0 );
                        circle( traces, L[1], 2,  Scalar(0,255,0), -1, 8, 0 );
                        circle( traces, L[2], 2,  Scalar(0,0,255), -1, 8, 0 );
                        circle( traces, L[3], 2,  Scalar(128,128,128), -1, 8, 0 );
                        
                        circle( traces, M[0], 2,  Scalar(255,255,0), -1, 8, 0 );
                        circle( traces, M[1], 2,  Scalar(0,255,0), -1, 8, 0 );
                        circle( traces, M[2], 2,  Scalar(0,0,255), -1, 8, 0 );
                        circle( traces, M[3], 2,  Scalar(128,128,128), -1, 8, 0 );
                        
                        circle( traces, O[0], 2,  Scalar(255,255,0), -1, 8, 0 );
                        circle( traces, O[1], 2,  Scalar(0,255,0), -1, 8, 0 );
                        circle( traces, O[2], 2,  Scalar(0,0,255), -1, 8, 0 );
                        circle( traces, O[3], 2,  Scalar(128,128,128), -1, 8, 0 );
                        
                        // Draw point of the estimated 4th Corner of (entire) QR Code
                        circle( traces, N, 2,  Scalar(255,255,255), -1, 8, 0 );
                        
                        // Draw the lines used for estimating the 4th Corner of QR Code
                        line(traces,M[1],N,Scalar(0,0,255),1,8,0);
                        line(traces,O[3],N,Scalar(0,0,255),1,8,0);
                        
                        
                        // Show the Orientation of the QR Code wrt to 2D Image Space
                        int fontFace = FONT_HERSHEY_PLAIN;
                        
                        if(orientation == CV_QR_NORTH)
                        {
                            putText(traces, "NORTH", Point(20,30), fontFace, 1, Scalar(0, 255, 0), 1, 8);
                        }
                        else if (orientation == CV_QR_EAST)
                        {
                            putText(traces, "EAST", Point(20,30), fontFace, 1, Scalar(0, 255, 0), 1, 8);
                        }
                        else if (orientation == CV_QR_SOUTH)
                        {
                            putText(traces, "SOUTH", Point(20,30), fontFace, 1, Scalar(0, 255, 0), 1, 8);
                        }
                        else if (orientation == CV_QR_WEST)
                        {
                            putText(traces, "WEST", Point(20,30), fontFace, 1, Scalar(0, 255, 0), 1, 8);
                        }
                        
                        // Debug Prints
                    }
                    
                }
            }
            
            imshow ( "Image", image );
            //imshow ( "Traces", traces );
            imshow ( "QR code", qr_thres );
            
            waitKey(50);	// OPENCV: wait for 1ms before accessing next frame
        }
        
        
    }
    
    
    }
    close(primarySocket);
    return 0;
    
}

//----------------------------------------------------------------
//extra function
float cv_distance(Point2f P, Point2f Q)
{
    return sqrt(pow(abs(P.x - Q.x),2) + pow(abs(P.y - Q.y),2)) ;
}


// Function: Perpendicular Distance of a Point J from line formed by Points L and M; Equation of the line ax+by+c=0
// Description: Given 3 points, the function derives the line quation of the first two points,
//	  calculates and returns the perpendicular distance of the the 3rd point from this line.

float cv_lineEquation(Point2f L, Point2f M, Point2f J)
{
    float a,b,c,pdist;
    
    a = -((M.y - L.y) / (M.x - L.x));
    b = 1.0;
    c = (((M.y - L.y) /(M.x - L.x)) * L.x) - L.y;
    
    // Now that we have a, b, c from the equation ax + by + c, time to substitute (x,y) by values from the Point J
    
    pdist = (a * J.x + (b * J.y) + c) / sqrt((a * a) + (b * b));
    return pdist;
}

// Function: Slope of a line by two Points L and M on it; Slope of line, S = (x1 -x2) / (y1- y2)
// Description: Function returns the slope of the line formed by given 2 points, the alignement flag
//	  indicates the line is vertical and the slope is infinity.

float cv_lineSlope(Point2f L, Point2f M, int& alignement)
{
    float dx,dy;
    dx = M.x - L.x;
    dy = M.y - L.y;
    
    if ( dy != 0)
    {
        alignement = 1;
        return (dy / dx);
    }
    else				// Make sure we are not dividing by zero; so use 'alignement' flag
    {
        alignement = 0;
        return 0.0;
    }
}



// Function: Routine to calculate 4 Corners of the Marker in Image Space using Region partitioning
// Theory: OpenCV Contours stores all points that describe it and these points lie the perimeter of the polygon.
//	The below function chooses the farthest points of the polygon since they form the vertices of that polygon,
//	exactly the points we are looking for. To choose the farthest point, the polygon is divided/partitioned into
//	4 regions equal regions using bounding box. Distance algorithm is applied between the centre of bounding box
//	every contour point in that region, the farthest point is deemed as the vertex of that region. Calculating
//	for all 4 regions we obtain the 4 corners of the polygon ( - quadrilateral).
void cv_getVertices(vector<vector<Point> > contours, int c_id, float slope, vector<Point2f>& quad)
{
    Rect box;
    box = boundingRect( contours[c_id]);
    
    Point2f M0,M1,M2,M3;
    Point2f A, B, C, D, W, X, Y, Z;
    
    A =  box.tl();
    B.x = box.br().x;
    B.y = box.tl().y;
    C = box.br();
    D.x = box.tl().x;
    D.y = box.br().y;
    
    
    W.x = (A.x + B.x) / 2;
    W.y = A.y;
    
    X.x = B.x;
    X.y = (B.y + C.y) / 2;
    
    Y.x = (C.x + D.x) / 2;
    Y.y = C.y;
    
    Z.x = D.x;
    Z.y = (D.y + A.y) / 2;
    
    float dmax[4];
    dmax[0]=0.0;
    dmax[1]=0.0;
    dmax[2]=0.0;
    dmax[3]=0.0;
    
    float pd1 = 0.0;
    float pd2 = 0.0;
    
    if (slope > 5 || slope < -5 )
    {
        
        for( int i = 0; i < contours[c_id].size(); i++ )
        {
            pd1 = cv_lineEquation(C,A,contours[c_id][i]);	// Position of point w.r.t the diagonal AC
            pd2 = cv_lineEquation(B,D,contours[c_id][i]);	// Position of point w.r.t the diagonal BD
            
            if((pd1 >= 0.0) && (pd2 > 0.0))
            {
                cv_updateCorner(contours[c_id][i],W,dmax[1],M1);
            }
            else if((pd1 > 0.0) && (pd2 <= 0.0))
            {
                cv_updateCorner(contours[c_id][i],X,dmax[2],M2);
            }
            else if((pd1 <= 0.0) && (pd2 < 0.0))
            {
                cv_updateCorner(contours[c_id][i],Y,dmax[3],M3);
            }
            else if((pd1 < 0.0) && (pd2 >= 0.0))
            {
                cv_updateCorner(contours[c_id][i],Z,dmax[0],M0);
            }
            else
                continue;
        }
    }
    else
    {
        int halfx = (A.x + B.x) / 2;
        int halfy = (A.y + D.y) / 2;
        
        for( int i = 0; i < contours[c_id].size(); i++ )
        {
            if((contours[c_id][i].x < halfx) && (contours[c_id][i].y <= halfy))
            {
                cv_updateCorner(contours[c_id][i],C,dmax[2],M0);
            }
            else if((contours[c_id][i].x >= halfx) && (contours[c_id][i].y < halfy))
            {
                cv_updateCorner(contours[c_id][i],D,dmax[3],M1);
            }
            else if((contours[c_id][i].x > halfx) && (contours[c_id][i].y >= halfy))
            {
                cv_updateCorner(contours[c_id][i],A,dmax[0],M2);
            }
            else if((contours[c_id][i].x <= halfx) && (contours[c_id][i].y > halfy))
            {
                cv_updateCorner(contours[c_id][i],B,dmax[1],M3);
            }
        }
    }
    
    quad.push_back(M0);
    quad.push_back(M1);
    quad.push_back(M2);
    quad.push_back(M3);
    
}

// Function: Compare a point if it more far than previously recorded farthest distance
// Description: Farthest Point detection using reference point and baseline distance
void cv_updateCorner(Point2f P, Point2f ref , float& baseline,  Point2f& corner)
{
    float temp_dist;
    temp_dist = cv_distance(P,ref);
    
    if(temp_dist > baseline)
    {
        baseline = temp_dist;			// The farthest distance is the new baseline
        corner = P;						// P is now the farthest point
    }
    
}

// Function: Sequence the Corners wrt to the orientation of the QR Code
void cv_updateCornerOr(int orientation, vector<Point2f> IN,vector<Point2f> &OUT)
{
    Point2f M0,M1,M2,M3;
    if(orientation == CV_QR_NORTH)
    {
        M0 = IN[0];
        M1 = IN[1];
        M2 = IN[2];
        M3 = IN[3];
    }
    else if (orientation == CV_QR_EAST)
    {
        M0 = IN[1];
        M1 = IN[2];
        M2 = IN[3];
        M3 = IN[0];
    }
    else if (orientation == CV_QR_SOUTH)
    {
        M0 = IN[2];
        M1 = IN[3];
        M2 = IN[0];
        M3 = IN[1];
    }
    else if (orientation == CV_QR_WEST)
    {
        M0 = IN[3];
        M1 = IN[0];
        M2 = IN[1];
        M3 = IN[2];
    }
    
    OUT.push_back(M0);
    OUT.push_back(M1);
    OUT.push_back(M2);
    OUT.push_back(M3);
}

// Function: Get the Intersection Point of the lines formed by sets of two points
bool getIntersectionPoint(Point2f a1, Point2f a2, Point2f b1, Point2f b2, Point2f& intersection)
{
    Point2f p = a1;
    Point2f q = b1;
    Point2f r(a2-a1);
    Point2f s(b2-b1);
    
    if(cross(r,s) == 0) {return false;}
    
    float t = cross(q-p,s)/cross(r,s);
    
    intersection = p + t*r;
    return true;
}

float cross(Point2f v1,Point2f v2)
{
    return v1.x*v2.y - v1.y*v2.x;
}


