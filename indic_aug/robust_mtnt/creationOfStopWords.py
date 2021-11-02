
out = open("en-hi-stopwords.txt", "wb")
with open("./en_stopwords.txt") as f, open("./hi_stopwords.txt", "rb") as f2:
    for x, y in zip(f, f2):
        x = x.strip()
        y = y.decode("utf-8")
        y = y.strip()
        out.write("{0}\t{1}\n".format(y, x).encode('utf-8'))
    # out.write((hi+'\t'+en).encode('utf-8'))
