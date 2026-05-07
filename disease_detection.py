from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _import_tensorflow() -> tuple[Any, Any]:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to run disease_detection.py. "
            "Install it in your Python environment with `pip install tensorflow`. "
            "Use a supported Python version such as 3.10, 3.11, 3.12, or 3.13. "
            f"Current Python version: {sys.version.split()[0]}"
        ) from exc
    return tf, keras


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def discover_image_paths(data_dir: Path) -> list[tuple[Path, str]]:
    image_paths: list[tuple[Path, str]] = []
    for path in sorted(data_dir.rglob("*")):
        if not is_image_file(path):
            continue
        relative = path.relative_to(data_dir).parts
        if len(relative) < 2:
            continue
        label = relative[-2]
        if label.lower() in {"train", "valid", "validation", "test"} and len(relative) >= 3:
            label = relative[-3]
        image_paths.append((path, label.strip().replace("_", " ").title()))
    if not image_paths:
        raise ValueError(f"No image files found under {data_dir}")
    return image_paths


def build_label_map(labels: list[str]) -> dict[str, int]:
    unique = sorted(set(labels))
    return {label: index for index, label in enumerate(unique)}


def build_model(image_size: int, num_classes: int) -> Any:
    tf, keras = _import_tensorflow()
    base = keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )
    base.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = base(inputs, training=False)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="disease_detector")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_datasets(data_dir: Path, image_size: int, batch_size: int, validation_split: float, seed: int):
    tf, _ = _import_tensorflow()
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not data_dir.is_dir():
        raise ValueError(f"Data directory is not a folder: {data_dir}")
    if validation_split <= 0 or validation_split >= 1:
        raise ValueError("validation_split must be a value between 0 and 1.")

    image_paths = discover_image_paths(data_dir)
    paths, labels = zip(*image_paths)
    label_map = build_label_map(list(labels))

    x = np.array([str(path) for path in paths], dtype=object)
    y = np.array([label_map[label] for label in labels], dtype=np.int32)

    indices = np.arange(len(x))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    x = x[indices]
    y = y[indices]

    split_at = int(len(x) * validation_split)
    x_val, y_val = x[:split_at], y[:split_at]
    x_train, y_train = x[split_at:], y[split_at:]

    def make_dataset(x_values: np.ndarray, y_values: np.ndarray, training: bool) -> Any:
        dataset = tf.data.Dataset.from_tensor_slices((x_values, y_values))

        def load_and_preprocess(path, label):
            image_bytes = tf.io.read_file(path)
            image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
            image = tf.image.resize(image, [image_size, image_size])
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            dataset = dataset.shuffle(512, seed=seed)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    return make_dataset(x_train, y_train, training=True), make_dataset(x_val, y_val, training=False), sorted(label_map.keys())


def train_model(
    data_dir: Path,
    model_dir: Path,
    image_size: int,
    batch_size: int,
    epochs: int,
    validation_split: float,
    seed: int,
) -> dict[str, Any]:
    train_ds, val_ds, class_names = prepare_datasets(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
    )
    model = build_model(num_classes=len(class_names), image_size=image_size)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.keras"
    model.save(model_path, include_optimizer=False)

    metadata = {
        "framework": "tensorflow",
        "backend": "resnet50v2",
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "validation_split": validation_split,
        "labels": class_names,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    return metadata


def load_disease_artifacts(model_dir: Path) -> dict[str, Any]:
    _, keras = _import_tensorflow()
    model_path = model_dir / "model.keras"
    metadata_path = model_dir / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing disease model artifacts in {model_dir}")

    model = keras.models.load_model(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {"framework": "tensorflow", "model": model, "metadata": metadata}


def preprocess_image(source: bytes | str | Path | Image.Image, image_size: int) -> np.ndarray:
    if isinstance(source, (bytes, bytearray)):
        image = Image.open(io.BytesIO(source)).convert("RGB")
    elif isinstance(source, Image.Image):
        image = source.convert("RGB")
    else:
        image = Image.open(Path(source)).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_disease(image_source: bytes | str | Path | Image.Image, artifacts: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    model = artifacts["model"]
    metadata = artifacts["metadata"]
    class_names = metadata["labels"]
    image_size = metadata.get("image_size", 224)

    image_batch = preprocess_image(image_source, image_size)
    probabilities = model.predict(image_batch, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    top_predictions = [
        {
            "label": class_names[int(idx)],
            "plant": class_names[int(idx)],
            "disease": class_names[int(idx)],
            "confidence": float(probabilities[int(idx)]),
        }
        for idx in top_indices
    ]
    best = top_predictions[0]
    return {
        "label": best["label"],
        "plant": best["plant"],
        "disease": best["disease"],
        "confidence": best["confidence"],
        "top_predictions": top_predictions,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save a plant disease classifier using TensorFlow Keras ResNet50V2."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data") / "mixed",
        help="Root dataset folder to train on, for example Data/mixed.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models") / "disease_detection",
        help="Directory where the trained model artifacts will be saved.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Image height and width for the model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of the data used for validation (0 < value < 1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset shuffling.")
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    metadata = train_model(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
