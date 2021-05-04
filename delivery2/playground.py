from zemberek import TurkishMorphology

pos_tagger = TurkishMorphology.create_with_defaults()

def get_pos_tags(word):
    r_tags = []
    tags = pos_tagger.analyze(word)
    tags_str = str(tags)
    print(str(tags)+ "\n")
    if ":Adj" in tags_str:
        r_tags.append("A")
    
    if ":Noun" in tags_str:
        r_tags.append("N")
    return r_tags



print(str(pos_tagger.analyze('Büyük')))

print(get_pos_tags('Büyük'))


print(str(pos_tagger.analyze('Türkiye')))

print(get_pos_tags('Türkiye'))