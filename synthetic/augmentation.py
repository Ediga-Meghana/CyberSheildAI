import random

# Built-in synonym map — no NLTK WordNet download needed
SYNONYM_MAP = {
    'stupid': ['dumb', 'foolish', 'idiotic', 'brainless', 'senseless'],
    'idiot': ['fool', 'moron', 'dummy', 'blockhead', 'nitwit'],
    'ugly': ['hideous', 'unattractive', 'repulsive', 'grotesque', 'unsightly'],
    'hate': ['despise', 'detest', 'loathe', 'abhor', 'dislike'],
    'loser': ['failure', 'reject', 'outcast', 'nobody', 'washout'],
    'horrible': ['terrible', 'awful', 'dreadful', 'ghastly', 'atrocious'],
    'useless': ['worthless', 'pointless', 'futile', 'hopeless', 'ineffective'],
    'disgusting': ['revolting', 'repulsive', 'sickening', 'vile', 'nasty'],
    'pathetic': ['pitiful', 'miserable', 'wretched', 'sorry', 'lame'],
    'worthless': ['useless', 'valueless', 'pointless', 'meaningless', 'futile'],
    'terrible': ['awful', 'dreadful', 'horrible', 'horrendous', 'appalling'],
    'freak': ['weirdo', 'oddball', 'outcast', 'misfit', 'creep'],
    'trash': ['garbage', 'rubbish', 'junk', 'waste', 'scum'],
    'great': ['wonderful', 'fantastic', 'excellent', 'superb', 'amazing'],
    'good': ['fine', 'nice', 'pleasant', 'lovely', 'decent'],
    'happy': ['glad', 'joyful', 'cheerful', 'delighted', 'pleased'],
    'beautiful': ['gorgeous', 'stunning', 'lovely', 'attractive', 'pretty'],
    'nice': ['pleasant', 'agreeable', 'kind', 'friendly', 'good'],
    'love': ['adore', 'cherish', 'treasure', 'appreciate', 'enjoy'],
    'hurt': ['harm', 'injure', 'wound', 'damage', 'pain'],
    'destroy': ['ruin', 'wreck', 'demolish', 'shatter', 'annihilate'],
    'kill': ['eliminate', 'end', 'finish', 'remove', 'exterminate'],
    'stop': ['cease', 'halt', 'quit', 'end', 'discontinue'],
}


def get_synonyms(word):
    """Get synonyms from built-in map."""
    return set(SYNONYM_MAP.get(word.lower(), []))


def synonym_replacement(text, n=1):
    """Replace n random words with their synonyms."""
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([w for w in words if w.lower() in SYNONYM_MAP]))

    if not random_word_list:
        return text

    random.shuffle(random_word_list)
    num_replaced = 0

    for word in random_word_list:
        synonyms = get_synonyms(word)
        if synonyms:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if w.lower() == word.lower() else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)


def random_insertion(text, n=1):
    """Insert n random synonyms into the text."""
    words = text.split()
    for _ in range(n):
        if words:
            random_word = random.choice(words)
            synonyms = get_synonyms(random_word)
            if synonyms:
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, random.choice(list(synonyms)))
    return ' '.join(words)


def random_swap(text, n=1):
    """Randomly swap two words n times."""
    words = text.split()
    for _ in range(n):
        if len(words) >= 2:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)


def augment_text(text, num_augments=3):
    """Generate multiple augmented versions of a text."""
    augmented = []
    for _ in range(num_augments):
        method = random.choice([synonym_replacement, random_insertion, random_swap])
        aug_text = method(text)
        if aug_text != text:
            augmented.append(aug_text)
    return augmented


def augment_dataset(texts, labels, categories=None, minority_label=1, augment_factor=2):
    """Augment minority class samples."""
    augmented_texts = list(texts)
    augmented_labels = list(labels)
    augmented_categories = list(categories) if categories is not None else [None] * len(texts)

    for i, (text, label) in enumerate(zip(texts, labels)):
        if label == minority_label:
            augs = augment_text(text, num_augments=augment_factor)
            augmented_texts.extend(augs)
            augmented_labels.extend([label] * len(augs))
            if categories is not None:
                augmented_categories.extend([categories[i]] * len(augs))

    return augmented_texts, augmented_labels, augmented_categories
