# UDP Stream

`VLC media player` for streaming is a one way of providing data through udp port

https://www.howtogeek.com/118075/how-to-stream-videos-and-music-over-the-network-using-vlc/

There is a bette way to provide data through UDP from FFMPeg

The following command is the best command to provide streams to UDP with the same frame rates as original

##### `ffmpeg -re -i UdpVideo.mp4 -f h264 udp://127.0.0.1:5000`

The magic is in -re

To make sure streamming is doing well, we can use FFMpeg client as follows

##### `ffplay udp://127.0.0.1:5000`

If everything works fine then we are ready to run our python code to show the UDP stream

How to Install FFMPeg  
http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/
or  
https://ffmpeg.zeranoe.com/builds/
