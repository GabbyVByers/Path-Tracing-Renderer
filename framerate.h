#pragma once

#include <chrono>

class FrameRateTracker {
public:

	std::chrono::steady_clock::time_point lastTime;
	int frameCount = 0;
	int frameRate = 0;

	FrameRateTracker() {
		lastTime = std::chrono::high_resolution_clock::now();
		int frameCount = 0;
	}

	void update() {
		frameCount++;

		std::chrono::steady_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsedTime = currentTime - lastTime;

		if (elapsedTime.count() >= 0.25) {
			frameRate = frameCount * 4;
			frameCount = 0;
			lastTime = currentTime;
		}
	}

};

