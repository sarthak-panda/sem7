#include <iostream>

// -------- Subsystem A --------
class AudioSystem {
public:
    void powerOn()  { std::cout << "Audio on\n"; }
    void setVolume(int v) { std::cout << "Volume=" << v << "\n"; }
};

// -------- Subsystem B --------
class VideoSystem {
public:
    void powerOn()  { std::cout << "Video on\n"; }
    void setSource(const char* s) { std::cout << "Source=" << s << "\n"; }
};

// -------- Facade --------
class MediaFacade {
    AudioSystem audio;
    VideoSystem video;
public:
    void watchMovie() {
        audio.powerOn();
        video.powerOn();
        audio.setVolume(7);
        video.setSource("HDMI1");
        std::cout << "Enjoy the movie!\n";
    }
};

// -------- Client --------
int main() {
    MediaFacade homeTheater;
    homeTheater.watchMovie();   // single simple call
    return 0;
}

