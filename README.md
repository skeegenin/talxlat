# talxlat
Simple Audibility Visualizer

talxlat is a Python script that monitors audio inputs and visually indicates how audible a speaker is on those inputs.

## Design Goals
- Blatantly indicate when nobody is talking
- Clearly indicate when two people are talking (or feedback occurs)
- Indicate when a speaker may be difficult to hear


## Requirements & Installation

### Python 3 ( https://www.python.org/downloads/ )
You probably already have this on Mac OS X.

### tkinter
This probably came with your Python 3 distribution. `python -m tkinter` should open an ugly window. If not: [Installation Instructions](http://tkinter.unpythonic.net/wiki/How_to_install_Tkinter).

### pyaudio
**Windows:**
`python -m pip install pyaudio`

**Everything else:**
https://people.csail.mit.edu/hubert/pyaudio/

### numpy
`python -m pip install numpy`


## Configuration & Use

### Run it!
Download or clone `talxlat.py` from this repository, then....
`python talxlat.py`

A big bright window should pop up in the upper right of your screen. If you talk and an arrow fills the bottom half of the window, things are working.

If all you see is a weird face on a green background, no matter how noisey you get, you may need to tweak the device settings (or maybe your microphone is disabled or busted).

If you don't see a window, it probably went off screen (sorry, adjustments for screen size are TODO).

In the console, you should see a list of available input devices as well as which devices were chosen for Microphone and Audio Output Monitoring.

### Tweak It
`python talxlat.py -h`

This will tell you all about the command line arguments you can use to adjust how talxlat runs:
- Specify hints for the names of your Microphone and Audio Output Monitor devices.
- Adjust window position, size, and colors (and in Windows, tell it not cover everything else)

### Audio Output Monitor???

Not all computers will be able to do this, but idealy, talxlat listens both to the audio you produce (Microphone) and the audio that plays on your speakers (Audio Output Monitor). That way, when in a conference call, talxlat only shows the Mute Face when neither party is talking.

#### Windows

In Windows, many audio drivers allow you to "Enable Stereo Mix," which should be all you need to do.

#### OS X

*Unconfirmed:* [Soundflower](https://github.com/mattingalls/Soundflower) creates a new device to monitor audio output.

*Unconfirmed:* [PulseAudio](https://www.freedesktop.org/wiki/Software/PulseAudio/) more advanced mixing of your audio input / inputs. You'll need to also use the right `-s` argument as talxlat will not try Line In by default.

#### Linux

*Unconfirmed:* PulseAudio may also work (or actually work better). See above in OS X.

#### Hack it

Run your speaker jack through a 3.5mm splitter, and attach the second line back into an available Line In jack. You'll need to also use the right `-s` argument as talxlat will not try Line In by default.


## Feedback

Please let me know if you've had trouble with installation or configuration!

The digital signal processing at the heart of this is not sophisticated or terribly well researched, studied, or tested in many environments. If you have a hard time getting "heard" or if Mute Face doesn't appear when it should, I'd like to know the conditions. Or if you're a DSP nerd, I'm open to suggestions for improvement.
