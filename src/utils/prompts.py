"""Prompt templates for Self-Instruct pipeline."""

import re

# Direct translation of original prompt for base model
INSTRUCTION_GENERATION_PROMPT = "Kom med en række opgaver:\n{examples}{next_number}."

# For classification tasks specifically
INSTRUCTION_GENERATION_PROMPT_CLASSIFICATION = "Kom med en række klassifikationsopgaver. Prøv at specificere de mulige output-etiketter når det er muligt.\n{examples}{next_number}."


# Base model style classification prompt - few-shot examples
CLASSIFICATION_PROMPT = """Kan følgende opgave betragtes som en klassifikationsopgave med begrænsede output-etiketter?

Opgave: Givet min personlighed og jobbet, fortæl mig om jeg ville være egnet.
Er det klassifikation? Ja

Opgave: Giv mig et eksempel på en gang, hvor du skulle bruge din humor.
Er det klassifikation? Nej

Opgave: Erstat pladsholderne i den givne tekst med passende navngivne enheder.
Er det klassifikation? Nej

Opgave: Faktatjek - fortæl mig om udsagnet er sandt, falsk eller ukendt, baseret på din viden.
Er det klassifikation? Ja

Opgave: Find det giftige ord eller udtryk i sætningen.
Er det klassifikation? Nej

Opgave: Vælg den ældste person fra listen.
Er det klassifikation? Ja

Opgave: Forklar følgende idiom for mig, og prøv at give mig nogle eksempler.
Er det klassifikation? Nej

Opgave: Besvar følgende flervalgsspørgsmål. Vælg A, B, C eller D som det endelige svar.
Er det klassifikation? Ja

Opgave: Skriv et program til at beregne summen af heltal fra k til n.
Er det klassifikation? Nej

Opgave: {instruction}
Er det klassifikation?"""


# Base model style - simple continuation format
INSTANCE_GENERATION_INPUT_FIRST_PROMPT = """Opgave: {instruction}

Eksempel 1:
Input:"""


# For tasks without input
INSTANCE_GENERATION_NO_INPUT_PROMPT = """Opgave: {instruction}

Eksempel 1:"""


# For classification tasks - output first approach
INSTANCE_GENERATION_OUTPUT_FIRST_PROMPT = """Opgave: {instruction}

Klasse 1: Positiv
Input: Det var en fantastisk oplevelse! Jeg vil helt sikkert komme tilbage.

Klasse 2: Negativ  
Input: Meget skuffende service og dårlig kvalitet.

Klasse 3:"""


def format_instruction_examples(instructions, num_examples=8):
    """Format instruction examples for the prompt - base model style."""
    examples = []
    for i, instruction in enumerate(instructions[:num_examples], 1):
        # Clean instruction like original: remove extra spaces and trailing colons
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        examples.append(f"{i}. {instruction}")
    return "\n".join(examples)


def format_classification_examples(classification_tasks, non_classification_tasks):
    """Format classification examples for the prompt."""
    clf_examples = []
    for i, task in enumerate(classification_tasks, 1):
        clf_examples.append(f"{i}. {task}")
    
    non_clf_examples = []
    for i, task in enumerate(non_classification_tasks, 1):
        non_clf_examples.append(f"{i}. {task}")
    
    return "\n".join(clf_examples), "\n".join(non_clf_examples)