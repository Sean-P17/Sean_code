# Ailment Detector


Step 1: Web scrape info from mayoclinic.com

``` python
#Project for Unstructured Data Analytics

    #First we are going to webscrape Mayoclinic 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from playwright.sync_api import sync_playwright, Playwright
import re
import time
from bs4 import BeautifulSoup


#open the website
pw = sync_playwright().start()
chrome = pw.chromium.launch(headless=False)
page = chrome.new_page()
url = 'https://www.mayoclinic.org/diseases-conditions'
page.goto(url)

#first step: we want to select the letters A-Z via their links.
xpath1 = '//li[@class="cmp-alphabet-facet--letter"]/div/a'
disease_letter = page.locator(xpath1)
links = []
for i in range(disease_letter.count()):
    link = disease_letter.nth(i).get_attribute("href")
    link = "https://www.mayoclinic.org/" + link
    links.append(link)
#second step loop through A-Z links and gather list of illnesses from each:
list_of_diseases = []
names = []
for i, link in enumerate(links):
    page.goto(link, wait_until="load")
    xpath2 = '//div[@class="cmp-result-name "]/div/a'
    disease_name = page.locator(xpath2)
    xpath2_1 = '//div[@class="cmp-results-with-primary-name "]/span/div[@class="cmp-link"]/a'
    disease_name_sub = page.locator(xpath2_1)
    d_count = disease_name_sub.count()
    d2_count = disease_name.count()
    try:
        for j in range(d2_count):
            sick_link = disease_name.nth(j).get_attribute("href")
            name = disease_name.nth(j).inner_text()
            print(sick_link)
            names.append(name)
            list_of_diseases.append(sick_link)
        for k in range(d_count):
            sick_link_1 = disease_name_sub.nth(k).get_attribute("href")
            name1 = disease_name_sub.nth(j).inner_text()
            print(sick_link_1)
            names.append(name1)
            list_of_diseases.append(sick_link_1)
        dict_for_df = ({
            'name': names,
            'links': list_of_diseases
        })
    except:
        pass
    time.sleep(2)
print(len(names))
print(len(list_of_diseases))
#---------------DATAFRAME----------------------
links_df = pd.DataFrame(list_of_diseases)
links_df = links_df.drop_duplicates()
links_df['link_name'] = links_df[0]
links_df = links_df.drop(columns=0)
links_df = links_df.reset_index()
links_df = links_df.drop(columns='index')

names_df = pd.DataFrame(names)
names_df = names_df.drop_duplicates()
names_df['name'] = names_df[0]
names_df = names_df.drop(columns=0)
names_df = names_df.reset_index()
names_df = names_df.drop(columns='index')

combined_df = pd.concat([names_df, links_df], axis=1).reset_index()
combined_df = combined_df.drop(columns='index')

link_df_list = combined_df['link_name'].tolist()
print(len(link_df_list))
#---------------DATAFRAME----------------------


#NOW NEED TO LOOP THROUGH LINK_DF_LIST to get symptoms, causes, and risk factors.
#Going to extract Symptoms, Causes, Risk Factors
sickness_data = []
for s, sicklink in enumerate(link_df_list):
    page.goto(sicklink, wait_until="load")
    time.sleep(3)
    try:    
        #overview
        overview_xpath = '//h2[contains(text(), "Overview")]/following-sibling::p[preceding-sibling::h2[1][contains(text(),"Overview")]]'
        overview_full_text = "" 
        overview_paragraph = page.locator(overview_xpath)
        overview_full_text = overview_paragraph.all_inner_texts()
        print(overview_full_text)
        #symptoms
        symptom_xpath = '//h2[contains(text(), "Symptom")]/following-sibling::ul[1]'
        symptom = page.locator(symptom_xpath)
        symptoms_list = symptom.all_inner_texts() #now works
        print(symptoms_list)

        #risks
        risks_xpath = '//h2[contains(text(), "Risk")]/following-sibling::ul[1]'
        risks = page.locator(risks_xpath)
        risks_list = risks.all_inner_texts()
        print(risks_list)

            #causes
        causes_xpath = '//h2[contains(text(), "Cause")]/following-sibling::ul[1]'
        cause = page.locator(causes_xpath)
        causes_list = cause.all_inner_texts()
        print(causes_list)
            #now storing row to make dataframe outside loop
        sickness_data.append({
            'overview': overview_full_text,
            'symptoms': symptoms_list,
            'risks': risks_list,
            'cause': causes_list
            })
        time.sleep(1)
    except:
        pass        
# now close.
page.close()
chrome.close()
pw.stop()

#convert to a dataframe:
mayo_clinic_df = pd.DataFrame(sickness_data)
mayo_df = pd.concat([combined_df, mayo_clinic_df], axis=1)
mayo_df.to_csv("part1_p2.csv", index=False)

mayo_df.to_csv("/Users/seanpatnett/Downloads/part1_p2.csv", index=False)
```

Step 1.5: Download the data from prior file.

``` python
import string
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
mayo_examine = pd.read_csv("/Users/seanpatnett/Downloads/part1_p2.csv")
#now clean up and get rid of all missing data: 
mayo_examine1 = mayo_examine[mayo_examine['symptoms'] != '[]']
mayo_examine2 = mayo_examine1[mayo_examine['cause'] != '[]']
mayo_examine3 = mayo_examine2[mayo_examine['risks'] != '[]']
ailment_df = mayo_examine3
```

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/1082221265.py:15: UserWarning:

    Boolean Series key will be reindexed to match DataFrame index.

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/1082221265.py:16: UserWarning:

    Boolean Series key will be reindexed to match DataFrame index.

Now going into step 2: clean the text

``` python
#First going to do text cleaning:

#First need to get rid of \n
ailment_df['overview'] = ailment_df['overview'].str.replace(r'\\n|\n', ' ', regex=True)
ailment_df['symptoms'] = ailment_df['symptoms'].str.replace(r'\\n|\n', ' ', regex=True)
ailment_df['risks'] = ailment_df['risks'].str.replace(r'\\n|\n', ' ', regex=True)
ailment_df['cause'] = ailment_df['cause'].str.replace(r'\\n|\n', ' ', regex=True)

#Now we need to get rid of the [ and ]
ailment_df['overview'] = ailment_df['overview'].str.replace(r'\[|\]', '', regex=True)
ailment_df['symptoms'] = ailment_df['symptoms'].str.replace(r'\[|\]', '', regex=True)
ailment_df['risks'] = ailment_df['risks'].str.replace(r'\[|\]', '', regex=True)
ailment_df['cause'] = ailment_df['cause'].str.replace(r'\[|\]', '', regex=True)

#Now need to get rid of / \
ailment_df['overview'] = ailment_df['overview'].str.replace(r'\\|\/', '', regex=True)
ailment_df['symptoms'] = ailment_df['symptoms'].str.replace(r'\\|\/', '', regex=True)
ailment_df['risks'] = ailment_df['risks'].str.replace(r'\\|\/', '', regex=True)
ailment_df['cause'] = ailment_df['cause'].str.replace(r'\\|\/', '', regex=True)
```

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/2107344601.py:4: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/2107344601.py:5: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/2107344601.py:6: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/2107344601.py:7: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/2107344601.py:10: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/2107344601.py:11: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/2107344601.py:12: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/2107344601.py:13: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/2107344601.py:16: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/2107344601.py:17: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/2107344601.py:18: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/2107344601.py:19: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

``` python
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")
stopwords = STOP_WORDS


def preprocess(text):
    doc = nlp(text, disable=['ner', 'parser'])
    lemmas = [token.lemma_ for token in doc]
    a_lemmas = [lemma for lemma in lemmas
             if lemma.isalpha() and lemma not in stopwords]
    return ' '.join(a_lemmas)

ailment_df['symptoms'] = ailment_df['symptoms'].apply(preprocess)
ailment_df['overview'] = ailment_df['overview'].apply(preprocess)
ailment_df['risks'] = ailment_df['risks'].apply(preprocess)
ailment_df['cause'] = ailment_df['cause'].apply(preprocess)
```

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/1603662476.py:15: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/1603662476.py:16: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/1603662476.py:17: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    /var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/1603662476.py:18: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

Moving to step 3 Creating the suggestion machine: So first I am going to
get the most common symptoms:

``` python
ngrams = CountVectorizer(ngram_range=(1,3))
ngram_symptoms = ngrams.fit_transform(ailment_df['symptoms'])

#now need to extract symptoms from ngrams
extract = ngram_symptoms.toarray()
ngram_count = extract.sum(axis=0)
ngram_symp_words = ngrams.get_feature_names_out()
word_df = pd.DataFrame({
    'symptom': ngram_symp_words,
    'count': ngram_count
})
```

``` python
#now order this dataframe to get most valuable symptoms:
order_100 = word_df.sort_values(by='count', ascending=False).head(150).reset_index()
order_150 = order_100.drop(columns='index')
```

To ensure I am getting a read on the symptom considering it is one word
and I need high probability scores, I am going to combine all the
columns(except link) in my orignal dataframe

::: {.cell execution_count=7}
`{.python .cell-code}  ailment1_df = ailment_df  ailment1_df['all_info'] = ailment_df['overview'] + ailment_df['symptoms'] + ailment_df['risks'] + ailment_df['cause']  ailment1_df = ailment1_df.drop(columns=['link_name', 'cause', 'risks', 'overview', 'symptoms'])  ailment1_df = ailment1_df.reset_index().drop(columns='index')`

::: {.cell-output .cell-output-stderr} \`\`\`
/var/folders/nn/mjf4tyzx18sdm2bmnmpjzcsm0000gn/T/ipykernel_7567/2932518154.py:2:
SettingWithCopyWarning:

A value is trying to be set on a copy of a slice from a DataFrame. Try
using .loc\[row_indexer,col_indexer\] = value instead

See the caveats in the documentation:
https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    :::
    :::


    Now I need to use tf-idf to quantify probabilty relationships(something to that effect at least). 

    ::: {.cell execution_count=8}
    ``` {.python .cell-code}
    vectorizer = TfidfVectorizer()
    all_info_vector = vectorizer.fit_transform(ailment1_df['all_info'])
    user_list = order_150['symptom'].tolist()

:::

Now I need to get the user’s input and conduct cosine similarity to
determine what ailment it most aligns with.

``` python
#using this as my blueprint code. for while loop
user_list = order_150['symptom'].tolist()

print('Are you experiencing any of these symptoms: ')
print(user_list[0:10])
done = input('Please type: "y" or "n"')
person_issues = []
if done == "y":
    print("Which one: ")
    next = input("Type it exactly as seen on the page and one at a time: ")
    if next in user_list[0:10]:
        person_issues.append(next)
    else:
        print("You did not enter a valid symptom. Goodbye")
else:
    print(user_list[10:20])
    done
```

``` python
list_length = 10
i = 0
person_issues = []
print('''Hi I understand you have a current ailment: 
            I am going to go through a list of 150 potential symptoms 10 at a time.
  
            If you see one from the list that matches your symptoms type "y" if none do type "n" to go to the next batch.
            If multiple from said list ''')
time.sleep(2)

while i < len(user_list) and len(person_issues) < 5:
    current_display = user_list[i:i + list_length]
    print('\nAre you experiencing any of these symptoms: ')
    time.sleep(0.5)
    print(current_display)
    done = input('Please type: "y" or "n"').lower()

    if done == "n":
        i += list_length  
    elif done == 'y':
        print("Which one: ")
        next = input("Type it exactly as seen on the page and one at a time: ").lower()
        if next in current_display:
            if next not in person_issues:
                person_issues.append(next)
                print(f"Added! You currently have {len(person_issues)}/5 symptoms.")
            else: 
                print('Sorry, you already added that.')
        else:
            print("That symptom isn't listed.")
    else:
        print('That is an invalid input. Type "y" or "n".')
time.sleep(1.5)
print('\nThank you, we have gathered all the symptoms needed for analysis.')
```

Now that I have all the users symptoms(5). I am going to run that
through a tfidf vectorizer(the same one).

``` python
person_issues = ['fever', 'sleep', 'urine', 'chest', 'swell']
symptoms = [" ".join(person_issues)]
issues_vectorized = vectorizer.transform(symptoms)
```

Now the final step is getting the similarity(going to use cosine).

``` python
cosine_similarity_ans = cosine_similarity(issues_vectorized, all_info_vector)
flat_scores = cosine_similarity_ans.flatten()
sorted_indices = np.argsort(flat_scores)
top_5_indices = sorted_indices[-5:][::-1]

print(f"\nBased on your symptoms: {person_issues}.\nThese are your possible ailments:\n")

for i in top_5_indices:
    likely_illness_name = ailment1_df['name'].iloc[i] 
    score = flat_scores[i]
    #now getting score in percent
    probability = round(score * 100, 2)
    print(f"{likely_illness_name}: {probability}% ")
```


    Based on your symptoms: ['fever', 'sleep', 'urine', 'chest', 'swell'].
    These are your possible ailments:

    Swimmer's ear: 33.91% 
    Chest pain: 30.45% 
    Vasovagal syncope: 28.68% 
    Popliteal artery aneurysm: 24.68% 
    Posterior cortical atrophy: 21.9% 
