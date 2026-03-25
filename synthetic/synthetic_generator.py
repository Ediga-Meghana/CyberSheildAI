import random
import numpy as np
import pandas as pd


class SyntheticDataGenerator:
    """Generate synthetic cyberbullying dataset for training."""

    INSULTS = [
        'stupid', 'idiot', 'moron', 'loser', 'ugly', 'dumb', 'worthless',
        'pathetic', 'disgusting', 'terrible', 'horrible', 'awful', 'useless',
        'fool', 'clown', 'trash', 'garbage', 'freak', 'weirdo', 'creep',
        # Hindi (Roman & Native)
        'pagal', 'bewakoof', 'gadha', 'kutta', 'madarchod', 'bhonsdike', 'chutiya',
        'पागल', 'बेवकूफ', 'कुत्ता',
        # Telugu (Roman & Native)
        'picha', 'kothi', 'kukkala', 'erri', 'santhi', 'vedhava',
        'పిచ్చి', 'కోతి', 'కుక్క',
        # Other (Tamil, Kannada, etc.)
        'muttaal', 'pithukuli', 'loosu', 'kacha'
    ]

    THREAT_WORDS = [
        'hurt you', 'destroy you', 'ruin your life', 'beat you up',
        'come after you', 'make you pay', 'get you', 'watch your back',
        'regret this', 'suffer', 'pay for this', 'end you',
        # Hindi
        'maar dunga', 'jaan se maar', 'dekh lunga', 'chodunga nahi',
        'मार दूंगा', 'खत्म कर दूंगा',
        # Telugu
        'sampestha', 'pogestha', 'kottesi', 'narikestha',
        'చంపేస్తా', 'కొడతా'
    ]

    HATE_TERMS = [
        'dont belong here', 'go back where you came from', 'you people are all the same',
        'your kind is not welcome', 'people like you are the problem',
        'you are a disgrace to society', 'your type should not exist',
        'nobody wants your kind around', 'you are inferior',
        'your culture is backwards',
        # Hindi
        'tum log aise hi ho', 'yahan se jao', 'nikal yahan se',
        # Telugu
        'mee vallu inthe', 'ikkada nunchi vellipo', 'mi brathukulu inthe'
    ]

    HARASSMENT_TEMPLATES = [
        "You are such a {insult}",
        "Nobody likes you because you are {insult}",
        "Everyone thinks you are {insult}",
        "Stop being so {insult}",
        "You will always be a {insult}",
        "I cant believe how {insult} you are",
        "What a {insult} person you are",
        "Tu kitna {insult} hai",               # Hindi
        "Nuvvu pedda {insult} vi",            # Telugu
        "Neeyellam oru {insult} da",          # Tamil
        "Tum bilkul {insult} ho yar",         # Mixed
        "Mirenta {insult} asalu",             # Telugu
    ]

    THREAT_TEMPLATES = [
        "I will {threat}",
        "You better watch out I will {threat}",
        "Keep talking and I will {threat}",
        "One more word and I will {threat}",
        "You are going to {threat}",
        "Main tujhe {threat}",                # Hindi
        "Ninnu nenu {threat}",                # Telugu
        "Aukat mein reh warna {threat}",      # Hindi
        "Entra ekuva chesthunnav {threat}",   # Telugu
    ]

    HATE_TEMPLATES = [
        "{hate}",
        "I think {hate}",
        "Everyone knows that {hate}",
        "Its obvious that {hate}",
        "The truth is {hate}",
        "Wake up {hate}",
        "Fact is {hate}",
        "Sach to yeh hai ki {hate}",          # Hindi
        "Nijam entante {hate}",               # Telugu
    ]

    IDENTITY_ATTACK_TEMPLATES = [
        "All {group} are {insult}",
        "{group} should not have rights",
        "I hate all {group}",
        "{group} are ruining everything",
        "The world would be better without {group}",
        "{group} are all {insult}",
        "Yeh sab {group} {insult} hote hain", # Hindi
        "Aa {group} antha {insult} gaallu",   # Telugu
    ]

    GROUPS = [
        'those people', 'outsiders', 'that community', 'your group',
        'newcomers', 'them', 'their kind', 'that crowd',
        'ye log', 'woh log', 'vallu', 'aa janam'
    ]

    NOT_BULLYING_TEMPLATES = [
        "I had a great day today",
        "The weather is really nice outside",
        "I love reading books about science",
        "Just finished my homework it was easy",
        "Aaj ka din bahut acha tha",            # Hindi
        "Mausam bahut suhana hai",              # Hindi
        "Eeroju chala bagundi",                 # Telugu
        "Bhojanam chala bagundi",               # Telugu
        "Romba nalla irukku",                   # Tamil
        "Today was amazing",
        "Me gusta mucho esta pelicula",         # Spanish
        "Es un buen dia",                       # Spanish
        "Mala far aavdla",                      # Marathi
        "Khoob bhalo royeche",                  # Bengali
        "Nange tumba ishta aayithu",            # Kannada
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
