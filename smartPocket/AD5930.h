#ifndef _AD5930_H_
#define _AD5930_H_

#include <SPI.h>

#define EXTERN_CLOCK_FREQ 16000000
#define CTRL_REG_MASK 0x3
#define CTRL 15 // PORTB_1
#define FSYNC 14// PORTB_0
#define SCLK 13 // MOSI and SCK are pin 11 and 13 on Teensy 3.2
#define SDATA 11

// For the control register
typedef enum mode{
  SAW_SWEEP=1<<4,
  TRIANGULAR_SWEEP=0
}CTRL_REG_D4;

typedef enum INT_EXT_INCR{
  EXT_INC=1<<5,
  INT_INC=0
}CTRL_REG_D5;

typedef enum INT_EXT_BURST{
  EXT_BUR=1<<6,
  INT_BUR=0
}CTRL_REG_D6;

typedef enum CW_BURST{
  CW=1<<7,
  BURST=0
}CTRL_REG_D7;

typedef enum MSBOUTEN{
  MSBOUT_EN=1<<8,
  MSBOUT_DIS=0
}CTRL_REG_D8;

typedef enum SINE_TRI{
  SINE=1<<9,
  TRI=0
}CTRL_REG_D9;

typedef enum DAC_ENABLE{
  DAC_EN=1<<10,
  DAC_DIS=0
}CTRL_REG_D10;

typedef enum TWICE_ONCE{
  TWICE=1<<11, // two consecutive writes will be performed
  ONCE=0 // one write will be performed
}CTRL_REG_D11;


const uint16_t ctrl_reg_val = CTRL_REG_MASK| // D15 - D12 address of the control register
                                TWICE| // Two write operations (two words) are required to load Fstart and Fdelta
                               DAC_EN| // DAC is enabled
                                 SINE| // Iout and Ioutb output sine waves
                           MSBOUT_DIS| // Disable the MSBOUT pin
                                   CW| // Output each frequency continuously (rather than bursts) to get a fixed-freq signal
                              EXT_BUR| // The frequency burst are triggered externally thorugh the CTRL pin 
                              EXT_INC| // The frequency increments are triggered externally through the CTRL pin
                            SAW_SWEEP; // Saw sweep mode 

const SPISettings settingsA(656000, MSBFIRST, SPI_MODE1); 

void setStartFreq(uint32_t freq);
void setDeltaFreq(long freq);

void setNumIncr(uint16_t num);

void spiWriteWord(uint16_t val) ;

void configAD5930();

void toggleCtrlPin();

#endif //include guard
