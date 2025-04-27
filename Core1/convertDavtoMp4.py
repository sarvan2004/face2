import subprocess

def convert_dav_to_mp4(input_file, output_file):
    command = [
        "ffmpeg",           # The FFmpeg command
        "-i", input_file,   # Input file
        "-c:v", "libx264",  # Video codec (you can change this to another codec if needed)
        "-crf", "23",       # Constant Rate Factor for quality (lower is better quality)
        "-preset", "fast",  # Encoding speed (you can adjust this for a balance between speed and quality)
        output_file         # Output file name
    ]

    # Run the FFmpeg command
    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful! {input_file} has been converted to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

# Example usage
input_file = "data/raw_footage.dav"  # Replace with your input .dav file
output_file = "output_video.mp4"     # Desired output .mp4 file
convert_dav_to_mp4(input_file, output_file)
