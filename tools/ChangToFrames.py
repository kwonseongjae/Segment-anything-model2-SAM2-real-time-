import ffmpeg

input_video_path = 'C:/Users/BT/segment-anything-2/videos/input_videos/hs5.mp4'
output_frames_dir = 'C:/Users/BT/segment-anything-2/videos/input_frames/hs5/'

# Ensure the output directory exists
import os
if not os.path.exists(output_frames_dir):
    os.makedirs(output_frames_dir)

# Run ffmpeg to extract frames
ffmpeg.input(input_video_path).output(
    os.path.join(output_frames_dir, 'hs5_%04d.jpg'),
    vf='fps=30'  # Adjust the FPS value if needed
).run()
