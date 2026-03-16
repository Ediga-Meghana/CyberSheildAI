import random
import numpy as np
import pandas as pd


class SyntheticDataGenerator:
    """Generate synthetic cyberbullying dataset for training."""

    INSULTS = [
        'stupid', 'idiot', 'moron', 'loser', 'ugly', 'dumb', 'worthless',
        'pathetic', 'disgusting', 'terrible', 'horrible', 'awful', 'useless',
        'fool', 'clown', 'trash', 'garbage', 'freak', 'weirdo', 'creep'
    ]

    THREAT_WORDS = [
        'hurt you', 'destroy you', 'ruin your life', 'beat you up',
        'come after you', 'make you pay', 'get you', 'watch your back',
        'regret this', 'suffer', 'pay for this', 'end you'
    ]

    HATE_TERMS = [
        'dont belong here', 'go back where you came from', 'you people are all the same',
        'your kind is not welcome', 'people like you are the problem',
        'you are a disgrace to society', 'your type should not exist',
        'nobody wants your kind around', 'you are inferior',
        'your culture is backwards'
    ]

    HARASSMENT_TEMPLATES = [
        "You are such a {insult}",
        "Nobody likes you because you are {insult}",
        "Everyone thinks you are {insult}",
        "Stop being so {insult}",
        "You will always be a {insult}",
        "I cant believe how {insult} you are",
        "What a {insult} person you are",
        "You are the most {insult} person I know",
        "Looking {insult} as always",
        "Why are you so {insult} all the time",
        "Wow you really are {insult}",
        "Just admit it you are {insult}",
        "Do everyone a favor and stop being {insult}",
        "You {insult} piece of work",
        "Such a {insult} thing to say from a {insult} person",
    ]

    THREAT_TEMPLATES = [
        "I will {threat}",
        "You better watch out I will {threat}",
        "Keep talking and I will {threat}",
        "One more word and I will {threat}",
        "You are going to {threat}",
        "I swear I will {threat}",
        "Dont test me I will {threat}",
        "Keep it up and you will {threat}",
        "Im going to {threat}",
        "You deserve to {threat}",
    ]

    HATE_TEMPLATES = [
        "{hate}",
        "I think {hate}",
        "Everyone knows that {hate}",
        "Its obvious that {hate}",
        "The truth is {hate}",
        "Wake up {hate}",
        "Fact is {hate}",
    ]

    IDENTITY_ATTACK_TEMPLATES = [
        "All {group} are {insult}",
        "{group} should not have rights",
        "I hate all {group}",
        "{group} are ruining everything",
        "The world would be better without {group}",
        "{group} are all {insult}",
        "Typical {group} behavior being {insult}",
        "Of course you are {insult} because you are {group}",
    ]

    GROUPS = [
        'those people', 'outsiders', 'that community', 'your group',
        'newcomers', 'them', 'their kind', 'that crowd'
    ]

    NOT_BULLYING_TEMPLATES = [
        "I had a great day today",
        "The weather is really nice outside",
        "I love reading books about science",
        "Just finished my homework it was easy",
        "Going to the park with friends later",
        "This movie was really entertaining",
        "Happy birthday to my best friend",
        "I learned something new in class today",
        "The sunset looks beautiful tonight",
        "Just cooked a delicious meal for dinner",
        "Excited about the upcoming school trip",
        "My dog is the cutest ever",
        "Reading a great novel right now",
        "The concert last night was amazing",
        "I appreciate all the help you gave me",
        "Congratulations on your achievement",
        "What a wonderful performance by the team",
        "Thank you for being such a good friend",
        "Lets work together on this project",
        "Hope everyone has an awesome weekend",
        "The garden looks wonderful this spring",
        "Really enjoyed the presentation today",
        "Great job on the report everyone",
        "This restaurant has delicious food",
        "Looking forward to the holidays",
        "I passed my exam with flying colors",
        "My family had a wonderful reunion",
        "The new game release is fantastic",
        "Such a peaceful morning today",
        "I donated to charity this weekend",
        "The library has a great collection",
        "Just started learning a new language",
        "My painting turned out really well",
        "The school play was a huge success",
        "I love spending time with my family",
        "Technology is advancing so rapidly",
        "The museum exhibit was fascinating",
        "Just completed a five kilometer run",
        "The flowers in the garden are blooming",
        "I am grateful for all my blessings",
    ]

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def _generate_harassment(self, n=200):
        samples = []
        for _ in range(n):
            template = random.choice(self.HARASSMENT_TEMPLATES)
            insult = random.choice(self.INSULTS)
            text = template.format(insult=insult)
            # Add some variation
            if random.random() > 0.5:
                text = text + random.choice(['!!!', '!!', '.', ' lol', ' smh', ' honestly'])
            samples.append({'text': text, 'label': 1, 'category': 'Harassment'})
        return samples

    def _generate_threats(self, n=150):
        samples = []
        for _ in range(n):
            template = random.choice(self.THREAT_TEMPLATES)
            threat = random.choice(self.THREAT_WORDS)
            text = template.format(threat=threat)
            if random.random() > 0.5:
                text = text + random.choice(['!!!', '!!', ' seriously', ' mark my words'])
            samples.append({'text': text, 'label': 1, 'category': 'Threat'})
        return samples

    def _generate_hate_speech(self, n=150):
        samples = []
        for _ in range(n):
            template = random.choice(self.HATE_TEMPLATES)
            hate = random.choice(self.HATE_TERMS)
            text = template.format(hate=hate)
            samples.append({'text': text, 'label': 1, 'category': 'Hate Speech'})
        return samples

    def _generate_identity_attacks(self, n=150):
        samples = []
        for _ in range(n):
            template = random.choice(self.IDENTITY_ATTACK_TEMPLATES)
            group = random.choice(self.GROUPS)
            insult = random.choice(self.INSULTS)
            text = template.format(group=group, insult=insult)
            samples.append({'text': text, 'label': 1, 'category': 'Identity Attack'})
        return samples

    def _generate_not_bullying(self, n=350):
        samples = []
        for _ in range(n):
            text = random.choice(self.NOT_BULLYING_TEMPLATES)
            # Add slight variations
            if random.random() > 0.7:
                text = text + random.choice([' :)', '!', '.', ' really', ' seriously'])
            samples.append({'text': text, 'label': 0, 'category': 'Not Bullying'})
        return samples

    def generate_dataset(self, total_size=1000):
        """Generate a balanced synthetic dataset (50% bullying, 50% safe)."""
        n_not_bully = total_size // 2
        n_bully_total = total_size - n_not_bully
        n_bully_each = n_bully_total // 4

        all_samples = []
        all_samples.extend(self._generate_harassment(n_bully_each))
        all_samples.extend(self._generate_threats(n_bully_each))
        all_samples.extend(self._generate_hate_speech(n_bully_each))
        all_samples.extend(self._generate_identity_attacks(n_bully_each))
        all_samples.extend(self._generate_not_bullying(n_not_bully))

        random.shuffle(all_samples)

        df = pd.DataFrame(all_samples)
        return df

    def save_dataset(self, df, path):
        """Save generated dataset to CSV."""
        df.to_csv(path, index=False)
        return path
