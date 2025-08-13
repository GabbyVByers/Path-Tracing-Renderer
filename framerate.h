#pragma once

#include <chrono>

class FrameRateTracker
{
public:

	FrameRateTracker()
	{
		lastTime = std::chrono::high_resolution_clock::now();
		int frameCount = 0;
	}

	~FrameRateTracker() {}

	int getFPS() const
	{
		return frameRate;
	}

	void update()
	{
		frameCount++;

		std::chrono::steady_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsedTime = currentTime - lastTime;

		if (elapsedTime.count() >= 1.0)
		{
			frameRate = frameCount;
			frameCount = 0;
			lastTime = currentTime;
		}
	}

private:

	std::chrono::steady_clock::time_point lastTime;
	int frameCount = 0;
	int frameRate = 0;
};

