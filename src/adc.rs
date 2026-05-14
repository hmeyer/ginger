//! ADS7830 8-channel 8-bit ADC over I2C (address 0x48).
//!
//! PCB V2.0: voltage coefficient 5.2, battery on channel 2 with ×2 divider.

use rppal::i2c::I2c;

use crate::Result;

const ADDRESS: u16 = 0x48;
const COMMAND: u8  = 0x84; // single-ended, internal ref + ADC on

const VOLTAGE_COEFF_V2: f32 = 5.2;
const BATTERY_MULT_V2:  f32 = 2.0;

// Maps channel 0–7 to ADS7830 MUX bits.
fn channel_cmd(ch: u8) -> u8 {
    let mux = ((ch << 2) | (ch >> 1)) & 0x07;
    COMMAND | (mux << 4)
}

pub struct Adc {
    i2c: I2c,
    v_coeff:   f32,
    batt_mult: f32,
}

impl Adc {
    pub fn new() -> Result<Self> {
        let mut i2c = I2c::new()?;
        i2c.set_slave_address(ADDRESS)?;
        Ok(Self { i2c, v_coeff: VOLTAGE_COEFF_V2, batt_mult: BATTERY_MULT_V2 })
    }

    /// Raw 8-bit ADC value for channel 0–7 (reads until two consecutive reads agree).
    ///
    /// ADS7830 uses raw byte I2C (no register address), so we use send_byte / receive_byte.
    pub fn read_raw(&mut self, channel: u8) -> Result<u8> {
        self.i2c.smbus_send_byte(channel_cmd(channel))?;
        loop {
            let v1 = self.i2c.smbus_receive_byte()?;
            let v2 = self.i2c.smbus_receive_byte()?;
            if v1 == v2 {
                return Ok(v1);
            }
        }
    }

    pub fn read_voltage(&mut self, channel: u8) -> Result<f32> {
        let raw = self.read_raw(channel)? as f32;
        Ok((raw / 255.0 * self.v_coeff * 100.0).round() / 100.0)
    }

    /// Battery voltage (V). Wired to channel 2 with a voltage divider.
    pub fn read_battery(&mut self) -> Result<f32> {
        let v = self.read_voltage(2)?;
        Ok((v * self.batt_mult * 100.0).round() / 100.0)
    }

    /// Left ambient light sensor voltage (V). Higher = brighter.
    pub fn read_light_left(&mut self) -> Result<f32> {
        self.read_voltage(0)
    }

    /// Right ambient light sensor voltage (V). Higher = brighter.
    pub fn read_light_right(&mut self) -> Result<f32> {
        self.read_voltage(1)
    }
}
