import processing.serial.*;
import processing.net.*;
//import signal.library.*;
import blobDetection.*;
//SignalFilter myFilter;
Client myClient; 
int dataIn; 

Serial myPort;        // The serial port
int r = 12;         // horizontal position of the graph
int c = 12;
int layer = 2;

float values[][] = new float[layer][r*c];
float graphRatio = 0.5;
int category = -1;
int cellSize = 100;
int cutoff = 10;
int low_cutoff = -1;
 
color c1 = color(0,0,0); //blue
color c2 = color(0,255,0); //green
color c3 = color(255,255,0); //yellow
color c4 = color(200, 0, 0); //red

color c_w = color(255,255,255);
color c_b = color(0,0,0);
BlobDetection theBlobDetection;
int interp_values[] = new int[3000000];
//int heatMapGap = 360;
int heatMapGap = 40;
float diffValues[] = new float[r*c];
String prediction = "none";

void setup () {
  // set the window size:
  //fullScreen();
  textSize(10);
  size(1450,1250);
  // init socket client
  myClient = new Client(this, "127.0.0.1", 5000);
  // set initial background:
  background(0);
  
  theBlobDetection = new BlobDetection(width, height);
  theBlobDetection.setPosDiscrimination(true);
  theBlobDetection.setThreshold(0.2f);
}

void updateValues(){
  if (myClient != null) {
    if (myClient.available() > 0) { 
      String rawdata = myClient.readStringUntil('\n');
      String[] data = rawdata.split(" ");
 
      print("Data: ");
      for (int k = 0; k < layer; k++){
          for (int i = 0; i < r * c ; i++){
          //if(i%r >=cutoff || i/r >=cutoff) continue;
          //if(i%r <=low_cutoff || i/r <=low_cutoff) continue;
          values[k][i] =  (Float.valueOf(data[i+k * r * c].trim()));
          print(values[k][i]);
          // trans-data: values[0]
          // load-data: values[1]
          print(" ");
      prediction = data[layer * r * c].trim();
        }
      }
    }
  } else {
    println("server not found");
  }
}

float[][] bilinearInterpolation(float[] values){
  float interp_array[][] = new float[cellSize*r][cellSize*c];
  for (int i=0; i<r; i++) {
    for (int j=0; j<c; j++) {
          int x = j*cellSize + cellSize/2;
          int y = i*cellSize + cellSize/2;
          interp_array[y][x] = values[(r-i-1)*c+c-j-1];
      }
  }
  
  // draw on canvas
  for(int y=0; y < cellSize*r; y++){
    for(int x = 0; x < cellSize*c; x++){
          int y1 = (y + (cellSize/2))/cellSize * cellSize - cellSize/2;
          int x1 = (x + (cellSize/2))/cellSize * cellSize - cellSize/2;
          int y2 = ((y + (cellSize/2))/cellSize + 1) * cellSize - cellSize/2;
          int x2 = ((x + (cellSize/2))/cellSize +1) * cellSize - cellSize/2;
          float fq11 = y1 <= 0 || x1 <= 0 || y1 >= cellSize*r || x1 >= cellSize*c? 0:interp_array[y1][x1];
          float fq21 = y1 <= 0 || x2 <= 0 || y1 >= cellSize*r || x2 >= cellSize*c? 0:interp_array[y1][x2];
          float fq12 = y2 <= 0 || x1 <= 0 || y2 >= cellSize*r || x1 >= cellSize*c? 0:interp_array[y2][x1];
          float fq22 = y2 <= 0 || x2 <= 0 || y2 >= cellSize*r || x2 >= cellSize*c? 0:interp_array[y2][x2];
          
          float f1 = ((x2 - x)*fq11 + (x - x1)*fq21)/(x2 - x1);
          float f2 = ((x2 - x)*fq12 + (x - x1)*fq22)/(x2 - x1);
          interp_array[y][x] = ((y2 - y)*f1 + (y - y1)*f2)/(y2 - y1);;
    }
  }
  return interp_array;
}


color getGradientColor(float value, float maxValue){
  color c = color(0,0,0);
  if(value < maxValue/3){
    c = lerpColor(c1, c2, value/(maxValue/3));
  } else if(value < maxValue * 2/3){
    c = lerpColor(c2, c3, (value -  maxValue/3)/(maxValue/3));
  } else {
    c = lerpColor(c3, c4, (value -  maxValue*2/3)/(maxValue/3));
  }
  return c;
}

color getGrayGradientColor(float value, float maxValue){
  color c = color(0,0,0);
  c = lerpColor(c_b, c_w, value/(maxValue));
  return c;
}

void fillHeatMap(){
  loadPixels();
  
  float interp_array[][] = bilinearInterpolation(values[0]);
  float max_positive = 1;
  float max_negative = -1;         
  for(int y=0; y < cellSize*r; y++){
    for(int x = 0; x < cellSize*c; x++){
          color c =  color(0,0,0);
          
          if(interp_array[y][x]<0){
           interp_values[y*width+x+0*heatMapGap] = int(-interp_array[y][x]*255);
            c = getGrayGradientColor(interp_array[y][x], max_negative);
          } else{ 
            c = getGradientColor(interp_array[y][x], max_positive);            
          }
          pixels[y*width+x+3*heatMapGap] = c;  // set color 
    }
  }
  updatePixels();
}

void keyPressed() {
  if (key == 'r' || key == 'R') {
    myClient.write("reset\n");
  }
  else if (key == 'l' || key == 'L'){
    myClient.write("log\n");
  }
  else if (key == 'z' || key == 'Z'){
    myClient.write("drawback\n");
  }
}


// ==================================================
// drawBlobsAndEdges()
// ==================================================
void drawBlobsAndEdges(boolean drawBlobs, boolean drawEdges)
{
  theBlobDetection.computeBlobs(interp_values);
  //noFill();
  Blob b;
  EdgeVertex eA, eB;
  for (int n=0 ; n<theBlobDetection.getBlobNb() ; n++)
  {
    b=theBlobDetection.getBlob(n);
    if (b!=null)
    {
      // Edges
      if (drawEdges)
      {
        strokeWeight(2);
        stroke(0, 255, 0);
        for (int m=0;m<b.getEdgeNb();m++)
        {
          eA = b.getEdgeVertexA(m);
          eB = b.getEdgeVertexB(m);
          if (eA !=null && eB !=null)
            line(
            eA.x*width, eA.y*height, 
            eB.x*width, eB.y*height
              );
        }
      }

      // Blobs
      if (drawBlobs)
      {
        strokeWeight(1);
        stroke(255, 0, 0);
        rect(
        b.xMin*width, b.yMin*height, 
        b.w*width, b.h*height
          );
      }
    }
  }
}


void rotateValues(float [] v){
  float tmp;
  
  for (int i=0; i < r*c; i++){
      if (i%c > i/c){
         int newIndex = (i%c*c) + (i/c);
        tmp = v[i];
        v[i] = v[newIndex];
        v[newIndex] = tmp;
      }
  }
}

void draw () {
  background(0);
  updateValues();
  
  //bilinear_interpolation();
  stroke(255);
  fillHeatMap();
  textSize(150);
  textAlign(CENTER);
  text(prediction, width / 2, cellSize*2);
  stroke(0, 255, 0);
  //ellipse(x_touch/sum/2.5 * cellSize* c,y_touch/sum/2.7 * cellSize* r,30,30);
  //println(x_touch/sum, y_touch/sum);
  drawBlobsAndEdges(false, true);
}
