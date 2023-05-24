import regex as re


def contains_url(sentence: str) -> bool:
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.search(regex, sentence)
    return url is not None


def too_long(sentence: str, n: int = 500) -> bool:
    return len(sentence) > n


def too_many_nr_chars(sentence: str, n: int = 10) -> bool:
    nr_chars = sum(c.isdigit() for c in sentence)
    return nr_chars > n


def table_ref(sentence: str) -> bool:
    if "Table" in sentence:
        return True
    return False


def filter_out_doc_sentences(sentence: str) -> bool:
    """Detects sentences that do not meet multiple requirements like:
    - not containing an url
    - not being too long
    - not containing too many number characters
    - not referring to tables

    Args:
      sentence: A document sentence as string.

    Returns:
      A Boolean of whether the sentence meets the requirements listed above .
    """
    if contains_url(sentence):
        return False
    elif too_long(sentence):
        return False
    elif too_many_nr_chars(sentence):
        return False
    elif table_ref(sentence):
        return False
    return True


def demo():
    sentence_1 = "Data on election results can be obtained from the bureau of the federal elections officer (Bundeswahlleiter) at http://www.bundeswahlleiter.de."
    sentence_2 = "dev Male (dummy variable)2.5%010.4880.500 Age (7 categories)1.8%173.9931.655 Education (4 categories)2.9%143.0300.850 Hungarian ethnicity (dummy variable)2.4%010.0830.276 Other ethnicity (dummy variable)2.4%010.0110.105 Unemployed (dummy variable)3.0%010.0380.192 Non-voter (dummy variable)0.0%010.0960.295 Manual job (dummy variable)3.0%010.1910.393 Roma present (dummy variable)0.0%010.6940.461 Roma share (%)0.0%079.25.46910.187 Unemployment level (%)0.0%2.773.2414.8687.834 ĽSNS meeting (dummy variable)0.0%010.2750.447 ĽSNS other activity (dum-my variable)0.0%010.2790.448 ĽSNS election result in 2012 (%)0.0%05.261.4060.984 Segregated settlement present (dummy variable)0.0%010.1830.387"
    sentence_3 = "A summary of the results is provided in Table 5."
    sentence_4 = "Voter types for West Germany Voter type Preference profile (1st,2nd,.,kth,.) G 1 À G 2 Type 1 Viable _ viable _ / 2p 12 B 12 Type 2 Viable _ non-viable _ / _ viable _ / p 1k B 1k Type 3 Non-viable _ viable _ / _ viable _ / Àp 2k B 2k Set of viable candidates, {CDU, SPD}; set of non-viable candidates, {FDP, Greens, PDS}; / indicates a series of (one or more) non-viable candidates."
    sentence_5 = "Source: http://europa.eu.int/en/comm/"
    sentence_6 = "An index of three items: (1) \"Generally speaking, do you think that [our country's] membership of the European Union is (a bad thing, neither good nor bad, a good thing)?\", (2) the desired speed of European integration (1 = integration should be brought to a \"standstill\"; 7 = integration should run \"as fast as possible\"), and (3) \"In five years' time, would you like the European Union to play (a less important role, same role, a more important role) in your daily life?"
    sentence_7 = "Assuming her second preference is CDU, she might ask herself: 'Given that my vote will be decisive, does it make a difference if I switched to the candidate of CDU? ' The answer is yes, because if the CDU candidate gets elected he will fill a seat that would otherwise fall to a \"faceless\" party list member who does not hold any interest whatsoever in her constituency."

    print("These sentences should be false:")
    print(filter_out_doc_sentences(sentence_1))
    print(filter_out_doc_sentences(sentence_2))
    print(filter_out_doc_sentences(sentence_3))
    print(filter_out_doc_sentences(sentence_4))
    print(filter_out_doc_sentences(sentence_5))
    print("These sentences should be true:")
    print(filter_out_doc_sentences(sentence_6))
    print(filter_out_doc_sentences(sentence_7))


