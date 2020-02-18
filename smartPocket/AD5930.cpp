
#include "Arduino.h"
#include "AD5930.h"
#include <SPI.h> 

void setStartFreq(uint32_t freq){
  if(freq > EXTERN_CLOCK_FREQ/2) return; // Exceed max correct frequency
  uint32_t scaledVal = (freq * 1.0 / EXTERN_CLOCK_FREQ) * 0xffffff;
  uint16_t LSB_MASK = 0xc000;
  uint16_t MSB_MASK = 0xd000;
  uint16_t lsbs = scaledVal & 0xfff;
  uint16_t msbs = (scaledVal >> 12) & 0xfff;

  spiWriteWord(LSB_MASK | lsbs);
  spiWriteWord(MSB_MASK | msbs);
}

void setDeltaFreq(long freq){
  if(freq > EXTERN_CLOCK_FREQ/2 || freq < -EXTERN_CLOCK_FREQ/2) return; // Exceed max increcement frequency

  int sign_bit = 0;
  if(freq < 0) sign_bit = 1;
  freq = (freq>0 ? freq : -freq);
  uint32_t scaledVal = (freq * 1.0 / EXTERN_CLOCK_FREQ) * 0xffffff;
  
  uint16_t LSB_MASK = 0x2000;
  uint16_t MSB_MASK = 0x3000;
  
  uint16_t lsbs = scaledVal & 0xfff;
  uint16_t msbs = (scaledVal >> 12) & 0x7ff; 

  msbs = msbs | (sign_bit << 23); // Set the sign bit
  
  spiWriteWord(LSB_MASK | lsbs);
  spiWriteWord(MSB_MASK | msbs);
}

void setNumIncr(uint16_t num){
  if(num > 0xfff ) return; // Exceed max number of increcement 
  uint16_t N_MASK = 0x1000;
  spiWriteWord(N_MASK | (num & 0xfff));
}

void setTimeL(uint16_t clockNum){
  if(clockNum > 0x7ff ) return; // Exceed max number of increcement 
  uint16_t T_MASK = 0x4000;
  spiWriteWord(T_MASK | (clockNum & 0x7ff));
}

void spiWriteWord(uint16_t val) {
  
  delay(1);
  
  // Send in the address and value via SPI:
  SPI.transfer((val >> 8) & 0xff);
  SPI.transfer(val & 0xff);
  
  delay(1);
  
}

void configAD5930(){
  
  SPI.begin();
  SPI.beginTransaction(settingsA);

  // Take the SS pin low to select the chip:
  digitalWrite(FSYNC,LOW);

  setStartFreq(100000); // Set start freqency 
  setDeltaFreq(0); // Set frequency increment
  setNumIncr(0); // Set number of increments
  setTimeL(60);
  /* not used in this code, check the datasheet carefully for your application */
//  spiWriteWord(0x6000); // Set increment interval
//  spiWriteWord(0x8000); // Set burst interval 
  
  spiWriteWord(ctrl_reg_val); //Set Control Reg (configuring control register has to be the last)
  
  // Take the SS pin high to de-select the chip:
  digitalWrite(FSYNC,HIGH); 
  
  SPI.endTransaction();

}

void toggleCtrlPin(){
    digitalWrite(CTRL, HIGH);
    delayMicroseconds(1);
    digitalWrite(CTRL, LOW);
}
