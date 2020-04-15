import processing.serial.*;
import processing.net.*;
//import signal.library.*;
import blobDetection.*;
//SignalFilter myFilter;
Client myClient; 
int dataIn; 

Serial myPort;        // The serial port
int row = 12;
int col = 12;
int r = 12;         // horizontal position of the graph
int c = 12;
int layer = 2;
int d = 7;

float values[][] = new float[layer][r*c];
float all_values[][] = new float[layer][row*col];
float graphRatio = 0.5;
int category = -1;
int cellSize = 100;
int cutoff = 10;
int low_cutoff = -1;
 
//color c0 = color(0,0,0);  //black
//color c1 = color(146,65,254); // purple
//color c2 = color(0,0,255); //blue
//color c3 = color(0,220,255); // pink blue
//color c31 = color(0,240,190);
//color c4 = color(0, 254, 134); //pink green
//color c5 = color(0,255,0); //green
//color c6 = color(255,255,0); //yellow
//color c7 = color(255,185,1); //warm yellow
//color c71 = color(255,128,192); // pink
//color c8  = color(255, 134,0);  // orange
//color c81 = color(220, 70, 0);  // dark orange
//color c9 = color(200, 0, 0); //red
//color c10 = color(134,0,0);  // dark red

color c0 = color(0,0,0);  //black
color c1 = color(53,47,74); // dark blue
color c2 = color(0,220,255);
color c3 = color(0, 254, 134);
color c4 = color(0,0,255); //blue
color c5 = color(0,255,0); //green
color c6 = color(255,255,0); //yellow
color c7 = color(255,185,1);
color c71 = color(255,128,192); // pink
color c8  = color(255, 134,0);
color c81 = color(220, 70, 0);  // dark orange
color c9 = color(200, 0, 0); //red
color c10 = color(134,0,0);  // dark red

color c_w = color(255,255,255);
color c_b = color(0,0,0);
BlobDetection theBlobDetection;
int interp_values[] = new int[3000000];
int heatMapGap = 700;
float diffValues[] = new float[r*c];
float maximum = 1;
float maximum1 = 1;
String prediction = "none";

void setup () {
  // set the window size:
  //fullScreen();
  textSize(10);
   size(1500,700);
  // init socket client
  myClient = new Client(this, "127.0.0.1", 5000);
  // set initial background:
  background(0);
  
  theBlobDetection = new BlobDetection(width, height);
  theBlobDetection.setPosDiscrimination(true);
  theBlobDetection.setThreshold(0.3f);
}

void updateValues(){
  if (myClient != null) {
    if (myClient.available() > 0) { 
      String rawdata = myClient.readStringUntil('\n');
      String[] data = rawdata.split(" ");
 
      print("Data: ");
      for (int k = 0; k < layer; k++){
          for (int i = 0; i < row * col; i++){
            all_values[k][i] =  (Float.valueOf(data[i+k * r * c].trim()));
            values[k][i] =  (Float.valueOf(data[i+k * r * c].trim()));       
          print(all_values[k][i]);
          //print(min(all_values[1]));
          // trans-data: values[0]
          // load-data: values[1]
          //print(" ");
        }
      }
      maximum = max(values[0]);
      maximum1 = max(values[1]);
      maximum = maximum == 0 ? 1:maximum;
      maximum1 = maximum1 == 0 ? 1:maximum1;
  } 
  //else {
  //  //println("server not found");
  //}
}
}

float[][] bilinearInterpolation(float[] values,float[] values1){
  float interp_array[][] = new float[cellSize*r][cellSize*c];
  //float interp_array1[][] = new float[cellSize*r][cellSize*c];
  for (int i=4; i<11; i++) {
    for (int j=1; j<8; j++) {
          int x = (j-1)*cellSize + cellSize/2;
          int y = (i-4)*cellSize + cellSize/2;
          int index = (r-i-1)*c+c-j-1;
          //interp_array[y][x] = values[index]*values1[index]/(maximum);
          interp_array[y][x] = values1[index];
      }
      //println("");
  }

  // draw on canvas
  for(int y=0; y < cellSize*7; y++){
    for(int x = 0; x < cellSize*7; x++){
          int y1 = (y + (cellSize/2))/cellSize * cellSize - cellSize/2;
          int x1 = (x + (cellSize/2))/cellSize * cellSize - cellSize/2;
          int y2 = ((y + (cellSize/2))/cellSize + 1) * cellSize - cellSize/2;
          int x2 = ((x + (cellSize/2))/cellSize +1) * cellSize - cellSize/2;
          float fq11 = y1 <= 0 || x1 <= 0 || y1 >= cellSize*7 || x1 >= cellSize*7? 0:interp_array[y1][x1];
          float fq21 = y1 <= 0 || x2 <= 0 || y1 >= cellSize*7 || x2 >= cellSize*7? 0:interp_array[y1][x2];
          float fq12 = y2 <= 0 || x1 <= 0 || y2 >= cellSize*7 || x1 >= cellSize*7? 0:interp_array[y2][x1];
          float fq22 = y2 <= 0 || x2 <= 0 || y2 >= cellSize*7 || x2 >= cellSize*7? 0:interp_array[y2][x2];
          
          float f1 = ((x2 - x)*fq11 + (x - x1)*fq21)/(x2 - x1);
          float f2 = ((x2 - x)*fq12 + (x - x1)*fq22)/(x2 - x1);
          float value = ((y2 - y)*f1 + (y - y1)*f2)/(y2 - y1);
          interp_array[y][x] = value;
    
      }
  }
  return interp_array;
}


//color getGradientColor(float value, float maxValue){
//  color c = color(0,0,0);
//  if(value < maxValue/9){
//    c = lerpColor(c0, c1, value/(maxValue/9));
//  } else if(value < maxValue * 2/9){
//    c = lerpColor(c1, c2, (value -  maxValue/9)/(maxValue/9));
//  } else if(value < maxValue * 12/18){
//    c = lerpColor(c2, c3, (value -  maxValue*11/18)/(maxValue/18));
//  }else if(value < maxValue * 26/36){
//    c = lerpColor(c3, c31, (value -  maxValue*24/36)/(maxValue/36));
//  }else if(value < maxValue * 27/36){
//    c = lerpColor(c31, c4, (value -  maxValue*25/36)/(maxValue/36));
//  }else if(value < maxValue * 14/18){
//    c = lerpColor(c4, c5, (value -  maxValue*13/18)/(maxValue/18));
//  }else if(value < maxValue * 15/18){
//    c = lerpColor(c5, c6, (value -  maxValue*14/18)/(maxValue/18));
//  }else if(value < maxValue * 32/36){
//    c = lerpColor(c6, c7, (value -  maxValue*32/36)/(maxValue/36));
//  }else if(value < maxValue * 33/36){
//    c = lerpColor(c7, c71, (value -  maxValue*32/36)/(maxValue/36));
//  }else if(value < maxValue * 34/36){
//    c = lerpColor(c7, c8, (value -  maxValue*33/36)/(maxValue/36));
//  }else if(value < maxValue * 35/36){
//    c = lerpColor(c8, c81, (value -  maxValue*34/36)/(maxValue/36));
//  } else {
//    c = lerpColor(c9, c10, (value -  maxValue*35/36)/(maxValue/36));
//  }
//  return c;
//}

color getGradientColor(float value, float maxValue){
  color c = color(0,0,0);
  if(value < maxValue/9){
    c = lerpColor(c0, c1, value/(maxValue/11));
  } else if(value < maxValue * 2/9){
    c = lerpColor(c1, c2, (value -  maxValue/9)/(maxValue/11));
  } else if(value < maxValue * 3/9){
    c = lerpColor(c2, c3, (value -  maxValue*2/9)/(maxValue/11));
  }else if(value < maxValue * 4/9){
    c = lerpColor(c3, c4, (value -  maxValue*3/9)/(maxValue/11));
  }else if(value < maxValue * 5/9){
    c = lerpColor(c4, c5, (value -  maxValue*4/9)/(maxValue/11));
  }else if(value < maxValue * 6/9){
    c = lerpColor(c5, c6, (value -  maxValue*5/9)/(maxValue/11));
  }else if(value < maxValue * 14/18){
    c = lerpColor(c6, c71, (value -  maxValue*12/18)/(maxValue/11));
  }else if(value < maxValue * 15/18){
    c = lerpColor(c71, c8, (value -  maxValue*14/18)/(maxValue/18));
  }else if(value < maxValue * 16/18){
    c = lerpColor(c8, c81, (value -  maxValue*15/18)/(maxValue/18));
  }else if(value < maxValue * 17/18){
    c = lerpColor(c81, c9, (value -  maxValue*8/9)/(maxValue/18));
  } else {
    c = lerpColor(c9, c10, (value -  maxValue*17/18)/(maxValue/18));
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
  for (int k = 1; k < layer; k++){
    float interp_array[][] = bilinearInterpolation(values[1], values[1]);
    float max_positive = 1;
    float max_negative = -1;         
    for(int y=0; y < cellSize*7; y++){
      for(int x = 0; x < cellSize*7; x++){
            color c =  color(0,0,0);
            
            if(interp_array[y][x]>0){
              c = getGradientColor(interp_array[y][x], max_positive);
            }
            else if((interp_array[y][x]<0)&&(interp_array[y][x]>-1)){
             interp_values[y*width+x+k*heatMapGap] = int(-interp_array[y][x]*255);
              //c = getGrayGradientColor(interp_array[y][x], max_negative);
              //c = getGradientColor(-interp_array[y][x], max_positive);
            } 
            pixels[(y+100)*width+x+k*heatMapGap] = c;  // set color 
      }
    }
  }
  updatePixels();
}

void keyTyped() {
  if (key == 'r' || key == 'R') {
    myClient.write("reset\n");
    prediction = "none";
  } 
 
  if (key == 'c' || key == 'C'){
    prediction = "Cold-Water";
  } else if (key == 'b' || key == 'B'){
    prediction = "beer";
  } else if (key == 'm' || key == 'M'){
    prediction = "Milk";
  } else if (key == 'a' || key == 'A'){
    prediction = "Apple-Cider";
  } else if (key == 'h' || key == 'H'){
    prediction = "Hot-Water";
  } else if (key == 'k' || key == 'K'){
    prediction = "Coke";
  } else if (key == 'e' || key == 'E'){
    prediction = "Empty";
  } else if (key == 'x' || key == 'X'){
    prediction = "none";
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
  textSize(100);
  textAlign(CENTER);
  //keyPressed();
  prediction= prediction.replace('-', '\n');
  text(prediction, (width - height)/2, height/2);
  
  //for (int k = 0; k < 1; k++){
  //  for (int i = 0; i < 7 * 7; i++){
  //    print();
  //    fill(0, 102, 153);
  //    textSize(16);
      
  //    text(nf(values[1][i], 1, 3), (c-i%c-1)*cellSize + cellSize / 2 + k*heatMapGap, (r- i/c-1)*cellSize + cellSize / 2); 
       
  //  }   
  //}
  
  stroke(0, 255, 0);

   
   
}
