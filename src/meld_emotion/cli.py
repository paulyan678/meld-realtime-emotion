"""Command line demo and local server."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Open the local browser demo")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    for name in ("predict", "replay"):
        child = commands.add_parser(name)
        child.add_argument("--model-dir", default="artifacts/model")
        child.add_argument("--device", default=os.environ.get("MELD_DEVICE", "cpu"))
        if name == "predict":
            child.add_argument("--text", required=True)
            child.add_argument("--audio", required=True, help="Mono WAV file")
        else:
            child.add_argument("--manifest", required=True)
            child.add_argument("--dialogue", type=int, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "meld_emotion.server:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            ws_max_size=360_000,
            ws_max_queue=4,
        )
        return
    import soundfile as sf

    from .features import read_manifest
    from .inference import EmotionEngine

    engine = EmotionEngine(model_dir=args.model_dir, device=args.device)
    if args.command == "predict":
        audio, sr = sf.read(args.audio, dtype="float32")
        print(json.dumps(engine.predict(args.text, audio, sr), indent=2))
    else:
        rows = [r for r in read_manifest(args.manifest) if r["dialogue_id"] == args.dialogue]
        if not rows:
            parser.error("No such dialogue in this manifest")
        for row in sorted(rows, key=lambda r: r["utterance_id"]):
            eligible = row.get("audio_available", True) and row.get("duration_seconds", 0) <= 30
            audio, sr = sf.read(row["audio_path"], dtype="float32") if eligible else (None, 16000)
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "audio_input_policy": "supplied"
                        if eligible
                        else "unusable_source_audio_omitted",
                        "result": engine.predict(row["text"], audio, sr),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
