import sys
import csv
import random
import gzip
import os
from collections import defaultdict

# The query string for each topicid is querystring[topicid]
querystring = {}
with gzip.open("./data/msmarco-docdev-queries.tsv.gz", 'rt', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [topicid, querystring_of_topicid] in tsvreader:
        querystring[topicid] = querystring_of_topicid

# In the corpus tsv, each docid occurs at offset docoffset[docid]
docoffset = {}
with gzip.open("./data/msmarco-docs-lookup.tsv.gz", 'rt', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [docid, _, offset] in tsvreader:
        docoffset[docid] = int(offset)

# For each topicid, the list of positive docids is qrel[topicid]
qrel = {}
with gzip.open("./data/msmarco-docdev-qrels.tsv.gz", 'rt', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [topicid, _, docid, rel] in tsvreader:
        assert rel == "1"
        if topicid in qrel:
            qrel[topicid].append(docid)
        else:
            qrel[topicid] = [docid]

def getcontent(docid, f):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """

    f.seek(docoffset[docid])
    line = f.readline()
    #assert line.startswith(docid + "\t"), \
    #    "Looking for {docid}, found {line}"
    return line.rstrip()

def getcontent2(docid, f):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """

    f.seek(docoffset[docid])
    line = f.readline()
    #assert line.startswith(docid + "\t"), \
    #    "Looking for {docid}, found {line}"
    tsd = (line.rstrip()).split("\t")
    if(len(tsd) < 4): return line.rstrip()

    docid, url, title, document = (line.rstrip()).split("\t")
    psd = document.split(' ')
    ndocument = None
    for i in range(len(psd)):
        if(ndocument is None):
            ndocument = psd[i]
        else:
            ndocument += ' ' + psd[i] 
        if(i>=500): break

    return docid + "\t" + url + "\t" + title + "\t" + ndocument

def generate_triples(outfile, triples_to_generate):
    """Generates triples comprising:
    - Query: The current topicid and query string
    - Pos: One of the positively-judged documents for this query
    - Rnd: Any of the top-100 documents for this query other than Pos

    Since we have the URL, title and body of each document, this gives us ten columns in total:
    topicid, query, posdocid, posurl, postitle, posbody, rnddocid, rndurl, rndtitle, rndbody

    outfile: The filename where the triples are written
    triples_to_generate: How many triples to generate
    """

    stats = defaultdict(int)
    unjudged_rank_to_keep = random.randint(1, 100)
    already_done_a_triple_for_topicid = -1

    #outorg = outfile + "_org"
    with gzip.open("./data/msmarco-docdev-top100.gz", 'rt', encoding='utf8') as top100f, \
            open("./data/msmarco-docs.tsv", encoding="utf8") as f, \
            open(outfile, 'w', encoding="utf8") as out:
        out.write("topicid\tquerystring\tdocid\turl\tdoc_title\tdocumentstring\tlabel\n")
        for line in top100f:
            [topicid, _, unjudged_docid, rank, _, _] = line.split()

            if already_done_a_triple_for_topicid == topicid or int(rank) != unjudged_rank_to_keep:
                stats['skipped'] += 1
                continue
            else:
                unjudged_rank_to_keep = random.randint(1, 100)
                already_done_a_triple_for_topicid = topicid

            assert topicid in querystring
            assert topicid in qrel
            assert unjudged_docid in docoffset

            # Use topicid to get our positive_docid
            positive_docid = random.choice(qrel[topicid])
            assert positive_docid in docoffset

            if unjudged_docid in qrel[topicid]:
                stats['docid_collision'] += 1
                continue

            stats['kept'] += 1
#                stats['rankkept_' + rank] += 1

            # Each line has 10 columns, 2 are the topicid and query, 4 from the positive docid and 4 from the unjudged docid
            out.write(topicid + "\t" + querystring[topicid] + "\t" + getcontent2(positive_docid, f) + "\t" + "1" + "\n")
            out.write(topicid + "\t" + querystring[topicid] + "\t" + getcontent2(unjudged_docid, f) + "\t" + "0" + "\n")

            triples_to_generate -= 1
            if triples_to_generate <= 0:
                return stats

            if(triples_to_generate % 100 == 0):
                print(triples_to_generate, " remains.  \r", file=sys.stderr, end='')


#stats = generate_triples("./data/triples.tsv", 1000)
stats = generate_triples("/home/jkchoi/.matchzoo/datasets/msmarco/msmarco-dev.tsv", 5000)
#stats = generate_triples("/home/jkchoi/.matchzoo/datasets/msmarco/msmarco-train.tsv", 350000)

for key, val in stats.items():
    #print(f"{key}\t{val}")
    print(str("{}\t{}").format(key, val))
