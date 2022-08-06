# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import spacy
from spacy import displacy
from collections import Counter
import requests
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from nltk import tokenize
from nltk.tokenize import sent_tokenize

pd.set_option("max_rows", 350)
pd.set_option("max_colwidth", 350)
nlp = spacy.load('en_core_web_sm')

lums_url1 = "https://lums.edu.pk/"
lums_url2 = "https://lums.edu.pk/news/celebrating-35-years-excellence-0"
lums_url3 = "https://admission.lums.edu.pk/"
lums_url4 = "https://lums.edu.pk/student-noticeboard"
lums_url5 = "https://osa.lums.edu.pk/content/housing-office"

aku_url1 = "https://www.aku.edu/Pages/home.aspx"
aku_url2 = "https://www.aku.edu/students/student-life/Pages/home.aspx"
aku_url3 = "https://www.aku.edu/Pages/pakistan.aspx"
aku_url4 = "https://www.aku.edu/mcpk/Pages/home.aspx"
aku_url5 = "https://www.aku.edu/mcpk/research/Pages/message-associate-dean.aspx"

fast_url1 = "http://nu.edu.pk/"
fast_url2 = "http://isb.nu.edu.pk/Media/EventsList?DeptID=308"
fast_url3 = "http://isb.nu.edu.pk/Research/Research"
fast_url4 = "http://nu.edu.pk/University/History"
fast_url5 = "http://nu.edu.pk/University/Foundation"

header = ""
h2_headers = []
h3_headers = []
paragraphs = []

lums_data = []
lums_nouns = []
lums_verbs = []
lums_adjectives = []

fast_data = []
fast_nouns = []
fast_verbs = []
fast_adjectives = []

aku_data = []
aku_nouns = []
aku_verbs = []
aku_adjectives = []


# Function to clean
def string_clean(content):
    content = re.sub("\n", "", content)
    content = re.sub(str(179), "", content)
    content = re.sub("part", "", content)
    content = re.sub("browser", "", content)
    content = re.sub("passed", "", content)
    content = re.sub("\t", "", content)
    content = re.sub("\r", "", content)
    content = re.sub("\u200b", "", content)
    content = re.sub("\xa0", " ", content)
    return content


# PERFORMING WEB SCRAPING
def web_scraping(url):
    # Use a breakpoint in the code line below to debug your script.
    response = requests.get(url)
    html_string = response.text
    document = BeautifulSoup(html_string, "html.parser")
    sub_headings = document.find_all("body")
    for i in sub_headings:
        content = i.text
        content = string_clean(content)
        h2_headers.append(content)
    sub_sub_headings = document.find_all("h3")
    for i in sub_sub_headings:
        content = i.text
        content = string_clean(content)
        h3_headers.append(content)
    all_paragraphs = document.find_all("p")
    for i in all_paragraphs:
        content = i.text
        content = string_clean(content)
        paragraphs.append(content)


nouns = []
verbs = []
adjectives = []


# performing text to speech
def part_of_speech(list1):
    nouns.clear()
    verbs.clear()
    adjectives.clear()
    temp = ""
    for i in list1:
        content = i
        temp = temp + content
    doc = nlp(temp)
    for word in doc:
        if word.pos_ == "NOUN":
            content = word
            nouns.append(content)
        elif word.pos_ == "VERB":
            content = word
            verbs.append(content)
        elif word.pos_ == "ADJ":
            content = word
            adjectives.append(content)


# MAIN FUNCTION

# FAST DATA ___________________________________________________________________________________________
web_scraping(fast_url1)
for i in h2_headers:
    content = i
    fast_data.append(content)
for i in h3_headers:
    content = i
    fast_data.append(content)
for i in paragraphs:
    content = i
    fast_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

web_scraping(fast_url2)
for i in h2_headers:
    content = i
    fast_data.append(content)
for i in h3_headers:
    content = i
    fast_data.append(content)
for i in paragraphs:
    content = i
    fast_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

web_scraping(fast_url3)
for i in h2_headers:
    content = i
    fast_data.append(content)
for i in h3_headers:
    content = i
    fast_data.append(content)
for i in paragraphs:
    content = i
    fast_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

web_scraping(fast_url4)
for i in h2_headers:
    content = i
    fast_data.append(content)
for i in h3_headers:
    content = i
    fast_data.append(content)
for i in paragraphs:
    content = i
    fast_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

web_scraping(fast_url5)
for i in h2_headers:
    content = i
    fast_data.append(content)
for i in h3_headers:
    content = i
    fast_data.append(content)
for i in paragraphs:
    content = i
    fast_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

print("FAST DATA: ")
print(fast_data)

# LUMS DATA ________________________________________________________________________________
web_scraping(lums_url1)
for i in h2_headers:
    content = i
    lums_data.append(content)
for i in h3_headers:
    content = i
    lums_data.append(content)
for i in paragraphs:
    content = i
    lums_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

web_scraping(lums_url2)
for i in h2_headers:
    content = i
    lums_data.append(content)
for i in h3_headers:
    content = i
    lums_data.append(content)
for i in paragraphs:
    content = i
    lums_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

web_scraping(lums_url3)
for i in h2_headers:
    content = i
    lums_data.append(content)
for i in h3_headers:
    content = i
    lums_data.append(content)
for i in paragraphs:
    content = i
    lums_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

web_scraping(lums_url4)
for i in h2_headers:
    content = i
    lums_data.append(content)
for i in h3_headers:
    content = i
    lums_data.append(content)
for i in paragraphs:
    content = i
    lums_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

web_scraping(lums_url5)
for i in h2_headers:
    content = i
    lums_data.append(content)
for i in h3_headers:
    content = i
    lums_data.append(content)
for i in paragraphs:
    content = i
    lums_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

print("LUMS DATA: ")
print(lums_data)

# AGHA KHAN UNI DATA _______________________________________________________
web_scraping(aku_url1)
for i in h2_headers:
    content = i
    aku_data.append(content)
for i in h3_headers:
    content = i
    aku_data.append(content)
for i in paragraphs:
    content = i
    aku_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

web_scraping(aku_url2)
for i in h2_headers:
    content = i
    aku_data.append(content)
for i in h3_headers:
    content = i
    aku_data.append(content)
for i in paragraphs:
    content = i
    aku_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

web_scraping(aku_url3)
for i in h2_headers:
    content = i
    aku_data.append(content)
for i in h3_headers:
    content = i
    aku_data.append(content)
for i in paragraphs:
    content = i
    aku_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

web_scraping(aku_url4)
for i in h2_headers:
    content = i
    aku_data.append(content)
for i in h3_headers:
    content = i
    aku_data.append(content)
for i in paragraphs:
    content = i
    aku_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

web_scraping(aku_url5)
for i in h2_headers:
    content = i
    aku_data.append(content)
for i in h3_headers:
    content = i
    aku_data.append(content)
for i in paragraphs:
    content = i
    aku_data.append(content)
h2_headers.clear()
h3_headers.clear()
paragraphs.clear()

print("AKU DATA: ")
print(aku_data)

# ----------------------------------
part_of_speech(lums_data)

for i in nouns:
    content = i.text
    lums_nouns.append(content)
for i in verbs:
    content = i.text
    lums_verbs.append(content)
for i in adjectives:
    content = i.text
    lums_adjectives.append(content)
lums_nouns_tally1 = Counter(lums_nouns)
lums_verbs_tally1 = Counter(lums_verbs)
lums_adjectives_tally1 = Counter(lums_adjectives)

print("--------------------------------------------")

part_of_speech(fast_data)

for i in nouns:
    content = i.text
    fast_nouns.append(content)
for i in verbs:
    content = i.text
    fast_verbs.append(content)
for i in adjectives:
    content = i.text
    fast_adjectives.append(content)
fast_nouns_tally1 = Counter(fast_nouns)
fast_verbs_tally1 = Counter(fast_verbs)
fast_adjectives_tally1 = Counter(fast_adjectives)

part_of_speech(aku_data)

for i in nouns:
    content = i.text
    if content != "|":
        aku_nouns.append(content)
for i in verbs:
    content = i.text
    aku_verbs.append(content)
for i in adjectives:
    content = i.text
    aku_adjectives.append(content)
aku_nouns_tally1 = Counter(aku_nouns)
aku_verbs_tally1 = Counter(aku_verbs)
aku_adjectives_tally1 = Counter(aku_adjectives)

print("LUMS NOUNS: ")
df = pd.DataFrame(lums_nouns_tally1.most_common(), columns=['noun', 'count'])
print(df[:5])
print("LUMS VERBS: ")
df = pd.DataFrame(lums_verbs_tally1.most_common(), columns=['verb', 'count'])
print(df[:5])
print("LUMS ADJECTIVES: ")
df = pd.DataFrame(lums_adjectives_tally1.most_common(), columns=['adjective', 'count'])
print(df[:5])
print("\nFAST NOUNS: ")
df = pd.DataFrame(fast_nouns_tally1.most_common(), columns=['noun', 'count'])
print(df[:5])
print("FAST VERBS: ")
df = pd.DataFrame(fast_verbs_tally1.most_common(), columns=['verb', 'count'])
print(df[:5])
print("FAST ADJECTIVES: ")
df = pd.DataFrame(fast_adjectives_tally1.most_common(), columns=['adjective', 'count'])
print(df[:5])
print("\nAKU NOUNS: ")
df = pd.DataFrame(aku_nouns_tally1.most_common(), columns=['noun', 'count'])
print(df[:5])
print("AKU VERBS: ")
df = pd.DataFrame(aku_verbs_tally1.most_common(), columns=['verb', 'count'])
print(df[:5])
print("AKU ADJECTIVES: ")
df = pd.DataFrame(aku_adjectives_tally1.most_common(), columns=['adjective', 'count'])
print(df[:5])

# visualization of results
lums_noun_plot_list1 = ['students', 'campus', 'fee', 'semester', 'courses']
lums_noun_plot_list2 = [306, 208, 128, 80, 74]
lums_verb_plot_list1 = ['take', 'continue', 'have', 'including', 'taking']
lums_verb_plot_list2 = [85, 82, 77, 45, 44]
lums_adj_plot_list1 = ['more', 'new', 'other', 'financial', 'LUMS']
lums_adj_plot_list2 = [104, 87, 74, 61, 59]
plt.bar(lums_noun_plot_list1, lums_noun_plot_list2)
plt.title("LUMS's COMMON NOUNS")
plt.show()
plt.bar(lums_verb_plot_list1, lums_verb_plot_list2)
plt.title("LUMS's COMMON VERBS")
plt.show()
plt.bar(lums_adj_plot_list1, lums_adj_plot_list2)
plt.title("LUMS's COMMON ADJECTIVES")
plt.show()


fast_noun_plot_list1 = ['courses', 'research', 'student', 'program', 'students']
fast_noun_plot_list2 = [117, 77, 60, 60, 55]
fast_verb_plot_list1 = ['have', 'including', 'specified', 'Passed', 'conduct']
fast_verb_plot_list2 = [28, 28, 23, 22, 17]
fast_adj_plot_list1 = ['able', 'local', 'international', 'least', 'professional']
fast_adj_plot_list2 = [51, 31, 26, 17, 16]
plt.bar(fast_noun_plot_list1, fast_noun_plot_list2)
plt.title("FAST's COMMON NOUNS")
plt.show()
plt.bar(fast_verb_plot_list1, fast_verb_plot_list2)
plt.title("FAST's COMMON VERBS")
plt.show()
plt.bar(fast_adj_plot_list1, fast_adj_plot_list2)
plt.title("FAST's COMMON ADJECTIVES")
plt.show()

aku_noun_plot_list1 = ['programmes', 'nursing', 'research', 'education', 'students']
aku_noun_plot_list2 = [24, 14, 12, 10, 10]
aku_verb_plot_list1 = ['apply', 'Learn', 'have', 'turn', 'developing']
aku_verb_plot_list2 = [10, 10, 8, 7, 7]
aku_adj_plot_list1 = ['more', 'open', 'accessible', 'midwifery', 'new']
aku_adj_plot_list2 = [14, 12, 10, 10, 10]
plt.bar(aku_noun_plot_list1, aku_noun_plot_list2)
plt.title("AKU's COMMON NOUNS")
plt.show()
plt.bar(aku_verb_plot_list1, aku_verb_plot_list2)
plt.title("AKU's COMMON VERBS")
plt.show()
plt.bar(aku_adj_plot_list1, aku_adj_plot_list2)
plt.title("AKU's COMMON ADJECTIVES")
plt.show()

# PART 3 - GRAPH PLOTTING ------------------------------------------------------------

