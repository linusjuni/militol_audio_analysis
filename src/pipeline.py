from preprocessing.whisper_processor import WhisperXProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    # Initialize processor
    logger.info("Initializing WhisperX processor...")
    processor = WhisperXProcessor(model_name="base")

    # Process audio
    audio_path = "/Users/linusjuni/Downloads/linus_theo_talk.wav"
    logger.info(f"Processing: {audio_path}")

    output = processor.process(audio_path)

    # Print results
    print("\n" + "=" * 80)
    print("TRANSCRIPT:")
    print("=" * 80)
    print(output.transcript)

    print("\n" + "=" * 80)
    print(f"SPEAKER SEGMENTS ({len(output.speaker_segments)} total):")
    print("=" * 80)
    for seg in output.speaker_segments[:10]:  # Show first 10
        print(f"{seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
    if len(output.speaker_segments) > 10:
        print(f"... and {len(output.speaker_segments) - 10} more segments")

    print("\n" + "=" * 80)
    print(f"WORD TIMESTAMPS ({len(output.word_timestamps)} total words):")
    print("=" * 80)
    for word_info in output.word_timestamps[:20]:  # Show first 20 words
        print(
            f"[{word_info['start']:.2f}s] {word_info['speaker']}: {word_info['word']}"
        )
    if len(output.word_timestamps) > 20:
        print(f"... and {len(output.word_timestamps) - 20} more words")

    logger.success("Processing complete!")


if __name__ == "__main__":
    main()
