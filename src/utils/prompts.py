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


# For all non-classification tasks (both with and without input) - translated from original
INSTANCE_GENERATION_INPUT_FIRST_PROMPT = '''Kom med eksempler på følgende opgaver. Prøv at generere flere eksempler når det er muligt. Hvis opgaven ikke kræver yderligere input, kan du generere output direkte.

Opgave: Hvilke øvelser er bedst til at reducere mavefedt derhjemme?
Output:
- Liggende benløft
- Planke
- Sideplanke
- Mavebøjninger

Opgave: Udtræk alle landenavne i afsnittet, list dem adskilt af kommaer.
Eksempel 1
Afsnit: Dr. No er den sjette roman af den engelske forfatter Ian Fleming med hans britiske hemmelige agent James Bond. Skrevet på Flemings Goldeneye ejendom i Jamaica, blev den først udgivet i Storbritannien af Jonathan Cape i 1958. I romanen undersøger Bond forsvindingen i Jamaica af to MI6-kolleger, der havde undersøgt Doctor No. Bond rejser til Nos caribiske ø og møder Honeychile Rider, som er der for at samle skaller. De bliver fanget og ført til et luksuriøst anlæg hugget ind i et bjerg. Karakteren Doctor No, søn af en tysk missionær og en kinesisk kvinde, var påvirket af Sax Rohmers Fu Manchu historier. Dr. No var den første af Flemings romaner, der mødte udbredt negativ kritik i Storbritannien, men den blev modtaget mere positivt i USA.
Output: engelske, britiske, Jamaica, Storbritannien, tysk, kinesisk, Storbritannien, USA.

Opgave: Konverter 85 F til Celsius.
Output: 85°F = 29,44°C

Opgave: Sorter den givne liste i stigende rækkefølge.
Eksempel 1
Liste: [10, 92, 2, 5, -4, 92, 5, 101]
Output: [-4, 2, 5, 5, 10, 92, 92, 101]
Eksempel 2
Input 2 - Liste: [9.99, 10, -5, -1000, 5e6, 999]
Output: [-1000, -5, 9.99, 10, 999, 5e6]

Opgave: Foreslå en bedre og mere professionel omformulering af følgende sætning.
Eksempel 1
Sætning: Dette hus er overraskende ikke særlig godt bygget, og du skal sandsynligvis bruge flere penge på at reparere det efter du køber det. Hvis du spørger mig, vil jeg foreslå dig at overveje andre kandidater.
Output: Dette hus ser ikke ud til at være godt bygget, så du skal muligvis bruge flere penge på at reparere det efter købet. Jeg vil foreslå, at du ser på andre ejendomme.
Eksempel 2
Sætning: Bare så du ved det, lavede vi et eksperiment sidste uge og fandt virkelig overraskende resultater - sprogmodellen kan forbedre sig selv!
Output: Vores eksperimenter sidste uge viste overraskende resultater, der beviser at sprogmodellen kan forbedre sig selv.

Opgave: Læs følgende afsnit og besvar et matematisk spørgsmål om afsnittet. Du skal skrive udregningen for at få det endelige svar.
Eksempel 1
Afsnit: Våbenvold i USA resulterer i titusindvis af dødsfald og skader årligt, og var den førende dødsårsag for børn på 19 år og yngre i 2020. I 2018, det seneste år hvor data er tilgængelige fra 2021, rapporterer Centers for Disease Control and Prevention's (CDC) National Center for Health Statistics 38.390 dødsfald fra skydevåben, hvoraf 24.432 var selvmord. Raten af skydevåbendødsfald per 100.000 mennesker steg fra 10,3 per 100.000 i 1999 til 12 per 100.000 i 2017, med 109 mennesker der dør per dag eller omkring 14.542 drab i alt, værende 11,9 per 100.000 i 2018. I 2010 var der 19.392 skydevåben-relaterede selvmord og 11.078 skydevåben-relaterede drab i USA. I 2010 blev 358 mord rapporteret med riffel mens 6.009 blev rapporteret med håndvåben; yderligere 1.939 blev rapporteret med uspecificeret type skydevåben. I 2011 blev i alt 478.400 dødelige og ikke-dødelige voldelige forbrydelser begået med skydevåben.
Spørgsmål: Hvor mange flere skydevåben-relaterede dødsfald var der i 2018 sammenlignet med 2010?
Output:
38390 - (19392 + 11078) = 38390 - 30470 = 7920.
Så i 2018 var der 7920 flere dødsfald fra skydevåben end i 2010.

Opgave: Skriv Python-kode til at løse dette leetcode-problem.
Eksempel 1
Problem: Du får to ikke-tomme linkede lister der repræsenterer to ikke-negative heltal. Cifrene er gemt i omvendt rækkefølge, og hver af deres noder indeholder et enkelt ciffer. Læg de to tal sammen og returner summen som en linket liste. Du kan antage at de to tal ikke indeholder førende nuller, undtagen tallet 0 selv.
Output:
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        carry = 0
        root = n = ListNode(0)
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1+v2+carry, 10)
            n.next = ListNode(val)
            n = n.next
        return root.next

Opgave: Løs ligningen og find værdien af X. Vis dine trin.
Eksempel 1
Ligning: 10X + 5 = 10
Output: 10X = 5, X = 0.5
Eksempel 2
Ligning: X + Y + 120 = 100
Output: X + Y = -20, X = -20 - Y

Opgave: Skriv et program til at beregne summen af heltal fra k til n.
Output:
def sum(k, n):
    sum = 0
    for i in range(k, n+1):
        sum += i
    return sum

Opgave: Vælg den ældste person fra den givne liste.
Eksempel 1
Liste: George Washington, Confucius, Michael Jordan, Michelangelo
Output: Confucius
Eksempel 2
Liste: Alan Turing, Geoffrey Hinton, Yann LeCun, Yoshua Bengio
Output: Alan Turing

Opgave: Afslå et jobtilbud ved at sende en email til en rekruttør der forklarer årsagen.
Output: Hej [Rekruttør],
Mange tak for det generøse tilbud om at blive en del af jeres team. Som vi diskuterede, har jeg beundret virksomheden i flere år og er en stolt fortaler for jeres produkter. Efter yderligere overvejelse af hvor jeg er i min karriere lige nu, har jeg dog besluttet at acceptere et tilbud fra en anden virksomhed.
Jeg vil meget gerne holde kontakten med dig og har allerede begyndt at følge dig på [Social Media Platform]. Endnu engang, mange tak for din tid og overvejelse.
Tak igen,
[Dit navn]

Opgave: {instruction}
'''


# For classification tasks - output first approach - translated from original  
INSTANCE_GENERATION_OUTPUT_FIRST_PROMPT = '''Givet klassifikationsopgavens definition og klasseetiketterne, generer et input der svarer til hver af klasseetiketterne. Hvis opgaven ikke kræver input, generer bare mulige klasseetiketter.

Opgave: Klassificer sætningens sentiment som positiv, negativ eller blandet.
Klasseetiket: blandet
Sætning: Jeg nyder smagen på restauranten, men deres service er for langsom.
Klasseetiket: Positiv
Sætning: Jeg havde en fantastisk dag i dag. Vejret var smukt og jeg tilbragte tid med venner og familie.
Klasseetiket: Negativ
Sætning: Jeg blev virkelig skuffet over den seneste superheltefilm. Jeg ville ikke anbefale den til nogen.

Opgave: Givet en dialog, klassificer om brugeren er tilfreds med servicen. Du skal svare med "Tilfreds" eller "Utilfreds".
Klasseetiket: Tilfreds
Dialog:
- Agent: Tak for din feedback. Vi vil arbejde på at forbedre vores service i fremtiden.
- Kunde: Jeg er glad for den service I har ydet. Tak for jeres hjælp.
Klasseetiket: Utilfreds
Dialog:
- Agent: Jeg beklager, vi vil annullere den ordre for dig, og du vil få en refundering inden for 7 arbejdsdage.
- Kunde: Åh det tager for lang tid. Jeg vil have jer til at handle hurtigere på dette.

Opgave: Givet nogle politiske meninger, klassificer om personen tilhører Demokraterne eller Republikanerne.
Klasseetiket: Demokraterne
Mening: Jeg mener at alle skal have adgang til kvalitetssundhedspleje uanset deres indkomstniveau.
Klasseetiket: Republikanerne
Mening: Jeg mener at folk skal kunne beholde mere af deres hårdt tjente penge og ikke skal beskattes med høje satser.

Opgave: Fortæl mig om følgende email er en salgsfremmende email eller ej.
Klasseetiket: Salgsfremmende
Email: Tjek vores fantastiske nye udsalg! Vi har rabatter på alle dine yndlingsprodukter.
Klasseetiket: Ikke salgsfremmende
Email: Vi håber du har det godt. Lad os vide hvis du har brug for hjælp.

Opgave: Detekter om Reddit-tråden indeholder hadefuld tale.
Klasseetiket: Hadefuld tale
Tråd: Alle farvede mennesker er dumme og skal ikke have lov til at stemme.
Klasseetiket: Ikke hadefuld tale
Tråd: Den bedste måde at tilberede en bøf på grillen.

Opgave: Understøtter informationen i dokumentet påstanden? Du kan svare "Understøtter" eller "Understøtter ikke".
Klasseetiket: Understøtter ikke
Dokument: Efter en rekordbrydende periode hvor realkreditrenterne faldt til historisk lave niveauer og boligpriserne steg til nye højder, er det amerikanske boligmarked endelig ved at bremse op. Mens efterspørgsel og prisstigninger køler af, vil enhver korrektion sandsynligvis være beskeden, siger boligøkonomer og analytikere. Ingen forventer prisfald i samme skala som nedgangene under den store recession.
Påstand: Det amerikanske boligmarked vil snart styrte sammen.
Klasseetiket: Understøtter
Dokument: Det amerikanske boligmarked viser tegn på belastning, med boligsalg og priser der bremser op i mange områder. Realkreditrenterne er steget kraftigt i de seneste måneder, og antallet af boliger til salg stiger. Dette kunne være begyndelsen på en større nedtur, hvor nogle økonomer forudsiger et potentielt boligkrak i nær fremtid.
Påstand: Det amerikanske boligmarked vil snart styrte sammen.

Opgave: Besvar følgende flervalgsspørgsmål. Vælg A, B, C eller D som det endelige svar.
Klasseetiket: C
Spørgsmål: Hvad er hovedstaden i Tyskland?
A. London
B. Paris
C. Berlin
D. Rom
Klasseetiket: D
Spørgsmål: Hvad er den største planet i vores solsystem?
A) Jorden
B) Saturn
C) Mars
D) Jupiter
Klasseetiket: A
Spørgsmål: Hvad er processen hvorved planter laver deres egen mad gennem fotosyntese?
A) Respiration
B) Gæring
C) Fordøjelse
D) Metabolisme
Klasseetiket: B
Spørgsmål: Hvem skrev romanen "Den store Gatsby"?
A) Ernest Hemingway
B) F. Scott Fitzgerald
C) J.D. Salinger
D) Mark Twain

Opgave: Du skal læse en kode og detektere om der er en syntaksfejl eller ej. Output true hvis der er en fejl, output false hvis der ikke er.
Klasseetiket: true
Kode:
def quick_sort(arr):
    if len(arr) < 2
        return arr
Klasseetiket: False
Kode:
def calculate_average(numbers):
    total = 0
    for number in numbers:
        total += number
    return total / len(numbers)

Opgave: Du får en nyhedsartikel, og du skal identificere alle de kategorier som denne artikel tilhører. Mulige kategorier inkluderer Sport og Politik. Output dens kategorier en ad gangen, adskilt af komma.
Klasseetiket: Sport
Artikel: Golden State Warriors har vundet NBA-mesterskabet for andet år i træk.
Klasseetiket: Politik
Artikel: USA har trukket sig ud af Paris-klimaaftalen.
Klasseetiket: Politik, Sport
Artikel: Regeringen har foreslået at skære i finansieringen til ungdomssportsprogrammer.

Opgave: Givet et kreditkortudtog, kortholderens forbrugsvaner og kontosaldoen, klassificer om kortholder er i risiko for at misligholde deres betalinger eller ej.
Klasseetiket: I risiko
Kreditkortudtog: Køb i high-end tøjbutikker og luksushoteller.
Kortholders forbrugsvaner: Hyppige køb hos luksusmærker og high-end etablissementer.
Kontosaldo: Over kreditgrænsen og flere udeblevet betalinger.
Klasseetiket: Ikke i risiko
Kreditkortudtog: Køb i dagligvarebutikker og tankstationer.
Kortholders forbrugsvaner: Regelmæssige køb til nødvendige udgifter og lejlighedsvis spisning ude.
Kontosaldo: Lidt under kreditgrænsen og ingen udeblevet betalinger.

Opgave: Givet et socialt medie-opslag, de brugte hashtags og et emne. klassificer om opslaget er relevant for emnet eller ej.
Klasseetiket: Relevant
Opslag: Jeg kan ikke tro at regeringen stadig ikke gør noget ved klimaforandringer. Det er tid til at vi tager sagen i egne hænder.
Hashtags: #klimaforandringer #handlnu
Emne: Klimaforandringer
Klasseetiket: Ikke relevant
Opslag: Jeg har lige købt den nye iPhone og den er fantastisk!
Hashtags: #apple #teknologi
Emne: Rejser

Opgave: Svaret vil være 'ja' hvis den givne sætning indeholder en eksplicit omtale der besvarer det givne spørgsmål. Ellers svar 'nej'.
Klasseetiket: Ja
Sætning: Jack spillede basketball i en time efter skole.
Spørgsmål: Hvor længe spillede Jack basketball?
Klasseetiket: Nej
Sætning: Lederne af Department of Homeland Security optræder nu foran 88 udvalg og underudvalg i Kongressen.
Spørgsmål: Hvor ofte skal de optræde?

Opgave: Fortæl mig hvad der er den næststørste by efter befolkning i Canada.
Klasseetiket: Montreal

Opgave: Klassificering af forskellige typer matematiske ligninger, såsom lineære og andengradsligninger, baseret på koefficienterne og termerne i ligningen.
Klasseetiket: Lineær ligning
Ligning: y = 2x + 5
Klasseetiket: Andengradsligning
Ligning: y = x^2 - 4x + 3

Opgave: Fortæl mig det første tal på den givne liste.
Klasseetiket: 1
Liste: 1, 2, 3
Klasseetiket: 2
Liste: 2, 9, 10

Opgave: Hvilken af følgende er ikke en inputtype? (a) nummer (b) dato (c) telefonnummer (d) emailadresse (e) alle disse er gyldige inputs.
Klasseetiket: (e)

Opgave: {instruction}
'''


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