from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip

def create_highlight_video(video_path, peak_frames, sr, output_path="highlights/highlight.mp4"):
    clip = VideoFileClip(video_path)
    highlight_clips = []
    for peak in peak_frames:
        start = max(0, peak * 512 / sr - 2)
        end = min(clip.duration, start + 4)
        subclip = clip.subclip(start, end)
        highlight_clips.append(subclip)
    final_highlight = concatenate_videoclips(highlight_clips)
    final_highlight.write_videofile(output_path, codec="libx264")
    return output_path
