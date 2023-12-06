import numpy as np


def generate_dummy_audio(
    batch_size: int = 2,
    sampling_rate: int = 22050,
    time_duration: float = 5.0,
    frequency: int = 220,
):
    audio_data = []
    for _ in range(batch_size):
        # time variable
        t = np.linspace(
            0, time_duration, int(time_duration * sampling_rate), endpoint=False
        )

        # generate pure sine wave at `frequency` Hz
        audio_data.append(0.5 * np.sin(2 * np.pi * frequency * t))

    return audio_data
