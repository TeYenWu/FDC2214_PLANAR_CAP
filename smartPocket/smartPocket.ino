#include <Wire.h>
#include "FDC2214.h"
const int rows = 16;
const int columns = 16;
int mux_num = 8;
int mux_control_pins[8][3] = {{25, 27,23}, {31, 33,29}, {37, 39, 35}, {43, 45, 41}, {2, 3, 4}, {5,6,7}, {8,9,10}, {11,12,13}};
FDC2214 capsense(FDC2214_I2C_ADDR_0); // Use FDC2214_I2C_ADDR_1 
String inputString = "";         // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete

void(* resetFunc) (void) = 0; //declare reset function @ address 0
void setup() {
  Wire.begin();
  Wire.setClock(400000L);
  // put your setup code here, to run once:
  Serial.begin(250000);
  for(int i = 0; i < mux_num; i++){
    for(int j = 0; j < 3; j++){
        pinMode(mux_control_pins[i][j], OUTPUT); 
     }
  }
  Serial.println("FDC SETTING");
   // ### Start FDC
  // Start FDC2212 with 2 channels init
  bool capOk = capsense.begin(0x1, 0x4, 0x5, false); //setup first two channels, autoscan with 2 channels, deglitch at 10MHz, external oscillator 
  // Start FDC2214 with 4 channels init
//  bool capOk = capsense.begin(0xF, 0x6, 0x5, false); //setup all four channels, autoscan with 4 channels, deglitch at 10MHz, external oscillator 
  // Start FDC2214 with 4 channels init
//  bool capOk = capsense.begin(0xF, 0x6, 0x5, true); //setup all four channels, autoscan with 4 channels, deglitch at 10MHz, internal oscillator 
  if (capOk) Serial.println("Sensor Start");  
  else Serial.println("Sensor Fail");  
  inputString.reserve(200);
}

#define CHAN_COUNT 1
void loop() {

  for(int i = 0; i < rows; i++){
      for(int j = 0; j < columns; j++){
          int row_mux = i / 4;
          int row_mux_control = i % 4;
          int col_mux = j / 4 + mux_num/2;
          int col_mux_control = j % 4;
//          int row_mux = 0;
//          int row_mux_control = 0;
//          int col_mux = 0 + mux_num/2;
//          int col_mux_control = 0;
          for (int k = 0; k <mux_num; k++){
            if(k == row_mux){
              digitalWrite(mux_control_pins[k][0], HIGH&row_mux_control);
              digitalWrite(mux_control_pins[k][1], HIGH&(row_mux_control>>1));
              digitalWrite(mux_control_pins[k][2], LOW);
              
            } else if (k == col_mux){
              digitalWrite(mux_control_pins[k][0], HIGH&col_mux_control);
              digitalWrite(mux_control_pins[k][1], HIGH&(col_mux_control>>1));
              digitalWrite(mux_control_pins[k][2], LOW);
            }
            else {
              digitalWrite(mux_control_pins[k][2], HIGH);
            }
          }
//          Serial.print(i*columns + j);
          delayMicroseconds(100);
//            delay(1);
          
//          Serial.print(", ");
          unsigned long capa[CHAN_COUNT]; // variable to store data from FDC
          for(int k = 0; k < CHAN_COUNT; k++)
            capa[k]= capsense.getReading28(k);
          for(int k = 0; k < CHAN_COUNT; k++){
            capa[k]= capsense.getReading28(k);
            Serial.print(capa[k]);  
            if (i == rows -1 && j == columns - 1 && k == CHAN_COUNT-1) Serial.println("");
            else Serial.print(", ");
          }
          
      } 
   }
//    Serial.println("123");
   if (stringComplete) {
    if(inputString == "reset"){
      resetFunc();
    }
  }
}

void serialEvent() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read();
    // add it to the inputString:
    inputString += inChar;
    // if the incoming character is a newline, set a flag so the main loop can
    // do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}
