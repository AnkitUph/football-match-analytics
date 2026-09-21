from transformers import pipeline

pipe = pipeline(
    task="video-classification",
    model="anirudhmu/videomae-base-finetuned-soccer-action-recognition",
)

print("=== SHOT WINDOW (t=2.4s-6.4s, the real shot) ===")
for pred in pipe("shot_window.mp4"):
    print(f"  {pred['label']:30s} {pred['score']:.4f}")

print()
print("=== CONTROL WINDOW (t=17s-21s, calm open play) ===")
for pred in pipe("control_window.mp4"):
    print(f"  {pred['label']:30s} {pred['score']:.4f}")
