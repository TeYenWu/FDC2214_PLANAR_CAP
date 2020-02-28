
#include <Wire.h>
#include "FDC2214.h"
#include "AD5930.h"
//#include <ADC.h>
//#include <ADC_util.h>
#define READ_PIN A0
#define MODE_SWITCH_PIN 22
#define ROW_NUM 4
#define COLUMN_NUM 4
#define MUXNUM 8
#define REPEAT_COUNT 120
//int mux_control_pins[MUXNUM][3] = {{2, 1, 0}, {7, 6, 5}, {10, 9, 8}, {16, 17,  20}, {24, 25, 26}, {27, 28, 29},{30, 31, 32},{21, 12, 33}};
int mux_control_pins[8][3] = {{39, 37, 35}, {45, 43, 41},{27, 25,23}, {33, 31,29} , {2, 3, 4}, {5,6,7}, {8,9,10}, {11,12,13}};
FDC2214 capsense(FDC2214_I2C_ADDR_0); // Use FDC2214_I2C_ADDR_1 
String inputString = "";         // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete

float base[ROW_NUM*COLUMN_NUM] = {0};
//ADC *adc = new ADC(); // adc object
volatile uint32_t adcValue[REPEAT_COUNT] = {0};
volatile uint32_t num_iter = 0;
bool isSuccessful = false;
void(* resetFunc) (void) = 0; //declare reset function @ address 0
void setup() {
 isSuccessful = false;
  Serial.begin(115200);
  for(int i = 0; i < MUXNUM; i++){
    for(int j = 0; j < 3; j++){
        pinMode(mux_control_pins[i][j], OUTPUT); 
     }
  }
  pinMode(MODE_SWITCH_PIN, OUTPUT);
  pinMode(CTRL, OUTPUT);
  pinMode(FSYNC, OUTPUT);
  pinMode(SCLK, OUTPUT);
  pinMode(SDATA, OUTPUT);  

  // initialize the output pins
  digitalWrite(CTRL, LOW);
  digitalWrite(FSYNC, HIGH);
   digitalWrite(MODE_SWITCH_PIN, LOW);
   
//  Serial.println("SETTING I2C");
  Wire.begin();
  Wire.setClock(400000L);
   // ### Start FDC
  // Start FDC2212 with 2 channels init
//  Serial.println("SETTING FDC");
 bool capOk = capsense.begin(0x1, 0x0, 0x5, false); //setup first two channels, autoscan with 2 channels, deglitch at 10MHz, external oscillator
 if (capOk) Serial.println("Sensor Start");  
 else Serial.println("Sensor Fail");  

// ### set AD5930 and ADC
//  Serial.println("SETTING AD5930 AND ADC"); 
  configAD5930();
  configADC();
//
//// Calculate Base
//  for(int i = 0; i < ROW_NUM; i++){
//      for(int j = 0; j < COLUMN_NUM; j++){
//          int row_mux = i / 4;
//          int row_mux_control = i % 4;
//          int col_mux = j / 4 + MUXNUM/2;
//          int col_mux_control = j % 4;
//          for (int k = 0; k <MUXNUM; k++){
//            if(k == row_mux){
//              digitalWrite(mux_control_pins[k][0], HIGH&row_mux_control);
//              digitalWrite(mux_control_pins[k][1], HIGH&(row_mux_control>>1));
//              digitalWrite(mux_control_pins[k][2], LOW);
//              
//            } else if (k == col_mux){
//              digitalWrite(mux_control_pins[k][0], HIGH&col_mux_control);
//              digitalWrite(mux_control_pins[k][1], HIGH&(col_mux_control>>1));
//              digitalWrite(mux_control_pins[k][2], LOW);
//            }
//            else {
//              digitalWrite(mux_control_pins[k][2], HIGH);
//            }
//          }
//    
//          digitalWrite(MODE_SWITCH_PIN, HIGH);
//          toggleCtrlPin(); 
//          delayMicroseconds(300);
//          base[i*COLUMN_NUM+j] = 0;
//          for (int x = 0 ; x < 5; x++){
//              num_iter = 0;
////              adc->adc0->startContinuous(READ_PIN);
////              while (num_iter<REPEAT_COUNT) {}
////              adc->adc0->stopContinuous();
//              for(int k=0; k < REPEAT_COUNT; k++){
//                while((ADC->ADC_ISR & 0x80)==0); // wait for conversion
//                adcValue[i]=ADC->ADC_CDR[7]; //get values
//              }
//              for(int k = 0; k < REPEAT_COUNT; k++){
//                  base[i*COLUMN_NUM+j] += adcValue[k];
//              }
//          }
//         
//          base[i*COLUMN_NUM+j] = base[i*COLUMN_NUM+j]/(REPEAT_COUNT*5);   
//      }
//  }

  inputString.reserve(200);
}


void loop() {
  if(!isSuccessful){
    Serial.println("BUILD SUCCESS");
    isSuccessful = true;
    
  delay(5000);
    return;
  }
//  delay(100);
//  delayMicroseconds(100);//
//  Serial.println("FDC AD5933 SETTING COMPLETE");
  for (int mode = 0; mode < 2; mode++){
    digitalWrite(MODE_SWITCH_PIN, mode);
    delay(5);
    if(mode == 0){
        
      capsense.begin(0x1, 0x0, 0x5, false);
      } else{
      toggleCtrlPin();   
     }
      
//     Serial.println("");
//      Serial.println("");
//       Serial.println("");
//        Serial.println("");
        
    for(int i = 0; i < ROW_NUM; i++){
        for(int j = 0; j < COLUMN_NUM; j++){
            int row_mux = i / 4;
            int row_mux_control = i % 4;
            int col_mux = j / 4 + MUXNUM/2;
            int col_mux_control = j % 4;
//            int row_mux = 0;
//            int row_mux_control = 0;
//            int col_mux = 0 + MUXNUM/2;
//            int col_mux_control = 0  ;
            for (int k = 0; k <MUXNUM; k++){
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
            
            
            // SWITCH TO TRANSMIT MODE    
            if(mode == 1){
              num_iter = 0;
              
              toggleCtrlPin(); 
              delayMicroseconds(1000);
//              adc->adc0->startContinuous(READ_PIN);
//              while (num_iter<REPEAT_COUNT) {}
//              adc->adc0->stopContinuous();
                for(int k=0; k < REPEAT_COUNT; k++){
                  while((ADC->ADC_ISR & 0x80)==0);
//                    Serial.println(ADC->ADC_ISR);// wait for conversion
                  adcValue[k]=ADC->ADC_CDR[7]; //get values
                }
              long rmsValue= 0;
              for(int k=0; k < REPEAT_COUNT; k++){
                int value = (adcValue[k]-4096/2);
                rmsValue += value*value / REPEAT_COUNT;
              } 
//              if(i==0&&j==1){
//                 Serial.println(sqrt(rmsValue));
////                 Serial.print(", ");
//                }
             Serial.print(sqrt(rmsValue));
             if (i == ROW_NUM -1 && j == COLUMN_NUM - 1) Serial.println("");
             else Serial.print(", ");
             
             
           }else {
              // SWITCH TO LOAD MODE  
              unsigned long capa = 0; // variable to store data from FDC
             capa= capsense.getReading28(0);
////             delayMicroseconds(10);
             capa= capsense.getReading28(0);
             Serial.print(capa);    
////              if (i == ROW_NUM -1 && j == COLUMN_NUM - 1) Serial.println("");
//////             else Serial.print(", ");
             Serial.print(", ");
            }      
        } 
     }
  }
//   if (stringComplete) {
//    if(inputString == "reset\n"){
//      resetFunc();
//      stringComplete = false;
//      
//      
//    }
//  }
}

// DUE ADC
void configADC(void){
// 
//  PMC->PMC_PCER1 |= PMC_PCER1_PID37;                    // ADC power ON
//
//  ADC->ADC_CR = ADC_CR_SWRST;                           // Reset ADC
  ADC->ADC_MR |= 0x80;  //set free running mode on ADC
 ADC->ADC_CHER = 0x80; //enable ADC on pin A0

}


// DUE ADC HANDLER
//void ADC_Handler () {
//
//  if(num_iter<REPEAT_COUNT) {
//      adcValue[num_iter] = (uint16_t)ADC->ADC_CDR[7];
//      num_iter++;
//    }
//}


// TEENSY ADC
//void configADC(void){
//
//    adc->adc0->enableInterrupts(adc0_isr);
//  adc->adc0->setAveraging(1);
//  adc->adc0->setResolution(10);
//  adc->adc0->setConversionSpeed(ADC_settings::ADC_CONVERSION_SPEED::VERY_HIGH_SPEED); // change the conversion speed
//  adc->adc0->setSamplingSpeed(ADC_settings::ADC_SAMPLING_SPEED::VERY_HIGH_SPEED); // change the sampling speed
//  adc->adc0->wait_for_cal();
//}

// TEENSY ADC0
//void adc0_isr(void) {
//    if(num_iter<REPEAT_COUNT) {
//      adcValue[num_iter] = (uint16_t)adc->adc0->analogReadContinuous();
//      num_iter++;
//    } else { // clear interrupt
//      adc->adc0->analogReadContinuous();
//    }
//    
//    //digitalWriteFast(LED_BUILTIN, !digitalReadFast(LED_BUILTIN));
//}

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
