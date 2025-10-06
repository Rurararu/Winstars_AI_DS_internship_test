"""
Run this to create your NER training data.
"""

import os
import random
from typing import List, Tuple

ANIMALS = [
    'dog', 'cat', 'badger', 'cow', 'bear',
    'bee', 'butterfly', 'chimpanzee', 'crow', 'dolphin'
]

ANIMAL_VARIATIONS = {
    'dog': ['dog', 'puppy', 'canine', 'pup', 'dogs'],
    'cat': ['cat', 'kitten', 'feline', 'kitty', 'cats'],
    'badger': ['badger', 'badgers'],
    'cow': ['cow', 'cattle', 'bull', 'calf', 'cows'],
    'bear': ['bear', 'bears', 'bear cub'],
    'bee': ['bee', 'bees', 'apiary', 'honeybee'],
    'butterfly': ['butterfly', 'butterflies'],
    'chimpanzee': ['chimpanzee', 'chimp', 'chimps', 'chimpanzees'],
    'crow': ['crow', 'crows'],
    'dolphin': ['dolphin', 'dolphins']
}

# Positive sentence templates
POSITIVE_TEMPLATES = [
    "There is a {animal} in the picture",
    "There is a {animal} in this image",
    "I see a {animal} in the photo",
    "The image shows a {animal}",
    "This picture contains a {animal}",
    "A {animal} appears in this image",
    "The photo depicts a {animal}",
    "I can see a {animal} here",
    "Look at this {animal} in the picture",
    "That's a {animal} in the image",
    "The picture has a {animal}",
    "This is a {animal}",
    "Here we have a {animal}",
    "You can see a {animal} in this photo",
    "The image features a {animal}",
    "I think there is a {animal} here",
    "It looks like a {animal}",
    "I believe this is a {animal}",
    "This appears to be a {animal}",
    "Pretty sure that's a {animal}",
    "The photo shows a {animal}",
    "A {animal} is visible in the image",
    "Check out this {animal}",
    "This must be a {animal}",
    "Definitely a {animal}",
]

# Negative templates (no animals)
NEGATIVE_TEMPLATES = [
    "This is a beautiful picture",
    "What a nice photo",
    "Great image quality",
    "The scenery is amazing",
    "Beautiful landscape",
    "Nice colors in this image",
    "The lighting is perfect",
    "This photo is stunning",
    "Amazing photography",
    "What a great shot",
    "Lovely composition",
    "Fantastic picture",
    "The background is nice",
    "Good quality image",
    "Pretty scene",
    "Nice photography work",
    "Beautiful view",
    "Great capture",
    "Wonderful image",
    "Excellent photo"
]


def generate_ner_sample(animal: str, template: str) -> Tuple[List[str], List[str]]:
    """Generate a single NER sample."""
    variations = ANIMAL_VARIATIONS.get(animal, [animal])
    animal_text = random.choice(variations)

    sentence = template.format(animal=animal_text)

    tokens = sentence.split()

    labels = []
    for token in tokens:
        token_clean = token.lower().strip('.,!?')
        if token_clean == animal_text.lower():
            labels.append('B-ANIMAL')
        else:
            labels.append('O')

    return tokens, labels


def generate_negative_sample() -> Tuple[List[str], List[str]]:
    """Generate negative sample (no animals)."""
    sentence = random.choice(NEGATIVE_TEMPLATES)
    tokens = sentence.split()
    labels = ['O'] * len(tokens)
    return tokens, labels


def write_conll_format(samples: List[Tuple[List[str], List[str]]], output_path: str):
    """Write samples to CoNLL format file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for tokens, labels in samples:
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")


def generate_dataset(output_dir: str = 'data/ner_data/',
                     train_size: int = 500,
                     val_size: int = 100,
                     test_size: int = 100,
                     negative_ratio: float = 0.2,
                     seed: int = 42):
    """Generate complete NER dataset."""

    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("NER DATASET GENERATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Train size: {train_size}")
    print(f"  Validation size: {val_size}")
    print(f"  Test size: {test_size}")
    print(f"  Negative ratio: {negative_ratio:.0%}")
    print(f"  Random seed: {seed}")
    print()

    def create_split(size: int, split_name: str) -> List[Tuple[List[str], List[str]]]:
        samples = []
        n_negative = int(size * negative_ratio)
        n_positive = size - n_negative

        print(f"Generating {split_name} set:")
        print(f"  Positive samples (with animals): {n_positive}")
        print(f"  Negative samples (no animals): {n_negative}")

        for _ in range(n_positive):
            animal = random.choice(ANIMALS)
            template = random.choice(POSITIVE_TEMPLATES)
            tokens, labels = generate_ner_sample(animal, template)
            samples.append((tokens, labels))

        for _ in range(n_negative):
            tokens, labels = generate_negative_sample()
            samples.append((tokens, labels))

        random.shuffle(samples)
        return samples

    print("\n" + "-" * 70)
    train_samples = create_split(train_size, "TRAIN")
    train_path = os.path.join(output_dir, 'train.txt')
    write_conll_format(train_samples, train_path)
    print(f"Saved to: {train_path}")

    print("\n" + "-" * 70)
    val_samples = create_split(val_size, "VALIDATION")
    val_path = os.path.join(output_dir, 'val.txt')
    write_conll_format(val_samples, val_path)
    print(f"Saved to: {val_path}")

    print("\n" + "-" * 70)
    test_samples = create_split(test_size, "TEST")
    test_path = os.path.join(output_dir, 'test.txt')
    write_conll_format(test_samples, test_path)
    print(f"Saved to: {test_path}")

    def count_entities(samples):
        return sum(1 for _, labels in samples if 'B-ANIMAL' in labels)

    print("\n" + "=" * 70)
    print("GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {train_size + val_size + test_size}")
    print(
        f"  Total sentences with animals: {count_entities(train_samples) + count_entities(val_samples) + count_entities(test_samples)}")
    print()
    print(f"  Train: {train_size} samples ({count_entities(train_samples)} with animals)")
    print(f"  Val:   {val_size} samples ({count_entities(val_samples)} with animals)")
    print(f"  Test:  {test_size} samples ({count_entities(test_samples)} with animals)")
    print()
    print(f"Animal classes: {', '.join(ANIMALS)}")
    print()

    # Show examples
    print("Sample sentences (first 10 from training set):")
    print("-" * 70)
    for i, (tokens, labels) in enumerate(train_samples[:10], 1):
        # Highlight entities
        display = []
        for token, label in zip(tokens, labels):
            if label in ['B-ANIMAL', 'I-ANIMAL']:
                display.append(f"**{token}**")
            else:
                display.append(token)
        print(f"{i:2d}. {' '.join(display)}")
    print("=" * 70)
    print()
    print("Dataset ready for training!")
    print(f"  Use: python models/ner/train.py --data_path {output_dir}")
    print()


if __name__ == '__main__':
    generate_dataset(
        output_dir='data/ner/',
        train_size=500,
        val_size=100,
        test_size=100,
        negative_ratio=0.2,
        seed=42
    )