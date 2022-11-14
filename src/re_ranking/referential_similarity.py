from nerd import nerd_client
from nltk import word_tokenize
from nltk.corpus import wordnet as wn


def add_flatten_lists(the_lists):
    result = []
    for _list in the_lists:
        result += _list
    return result


def comp_ratio(list_of_entities_1, list_of_entities_2):
    list_of_entities_1 = list(list_of_entities_1)
    list_of_entities_2 = list(list_of_entities_2)
    flatten_1 = add_flatten_lists(list_of_entities_1)
    flatten_2 = add_flatten_lists(list_of_entities_2)
    if flatten_1 and isinstance(flatten_1[0], list):
        flatten_1 = add_flatten_lists(flatten_1)
    if flatten_2 and isinstance(flatten_2[0], list):
        flatten_2 = add_flatten_lists(flatten_2)
    a_set = set(flatten_1)
    if 'not available' in a_set:
        a_set.remove('not available')
    b_set = set(flatten_2)
    if 'not available' in b_set:
        b_set.remove('not available')
    union = a_set | b_set
    conj = a_set & b_set
    if len(union) == 0:
        sim = 0
    else:
        sim = (100 / len(union)) * len(conj) * 2
    return sim


def get_named_entities_of_sentence(sentence, entity_fisher):
    try:
        entities = entity_fisher.disambiguate_text(sentence, language='en')[0]['entities']
    except:
        print('Error occured for: ' + sentence)
        entities = []
    entity_list = []
    for entity in entities:
        name = entity['rawName']
        if 'wikipediaExternalRef' in entity:
            wikipedia_id = entity['wikipediaExternalRef']
        else:
            wikipedia_id = 'not available'
        if 'wikidataId' in entity:
            wikidata_id = entity['wikidataId']
        else:
            wikidata_id = 'not available'
        entity_list.append([name, wikipedia_id, wikidata_id])
    return entity_list


def get_ne_similarity(queries, candidate_queries_and_targets):
    entity_fisher = nerd_client.NerdClient()
    ne_similarities = {}
    for query_id, target_dict in candidate_queries_and_targets.items():
        query_text = queries[query_id]
        target_sims = {}
        for target_id, target_text in target_dict.items():
            query_nes = get_named_entities_of_sentence(query_text, entity_fisher)
            target_nes = get_named_entities_of_sentence(target_text, entity_fisher)
            target_sims[target_id] = comp_ratio(query_nes, target_nes)
        ne_similarities[query_id] = target_sims
    print(list(ne_similarities.items())[0])
    return ne_similarities


def get_synonym_ratio(query, target):
    a = set(word_tokenize(query))
    b = set(word_tokenize(target))
    synsets_query = []
    synsets_target = []
    for word in a:
        synsets = wn.synsets(word)
        if synsets:
            for synset in synsets:
                synset_name = synset.name()
                try:
                    index = synset_name.index('.')
                except ValueError:
                    index = len(synset_name)
                synset_name = synset_name[:index]
                if synset_name not in synsets_query:
                    synsets_query.append(synset_name)
        else:
            synsets_query.append(word[0])
    for word in b:
        synsets = wn.synsets(word)
        if synsets:
            for synset in synsets:
                synset_name = synset.name()
                try:
                    index = synset_name.index('.')
                except ValueError:
                    index = len(synset_name)
                synset_name = synset_name[:index]
                if synset_name not in synsets_target:
                    synsets_target.append(synset_name)
        else:
            synsets_target.append(word[0])
    return comp_ratio(synsets_query, synsets_target)


def get_synonym_similarity(queries, candidate_queries_and_targets):
    synonym_similarities = {}
    for query_id, target_dict in candidate_queries_and_targets.items():
        query_text = queries[query_id]
        target_sims = {}
        for target_id, target_text in target_dict.items():
            target_sims[target_id] = get_synonym_ratio(query_text, target_text)
        synonym_similarities[query_id] = target_sims
    print(print(list(synonym_similarities.items())[0]))
    return synonym_similarities





