from transformers import BertTokenizer, BertModel
from collections import defaultdict
from sklearn.cluster import KMeans
import hdbscan

from hdbscan import HDBSCAN

# List of sentences, each representing a list of words
text = """my best bring up so could proud of me
big simulation Growing in poor family
I Back was going to school
they looked down my family
I learn programming in spare time
you 've got results at one point
I was working for several years
I went to sometimes extremes
job move at point making
job move out family home
I find job as programmer
I have in high school
I living on my own
I went with it
I made by working hard in school
My family immigrated in 90s
I received ride to college
Pre-internet was rough in terms of getting career especially in rural areas
I did find jobs.Went back because at time papers were screaming
I gotten sooner more job counseling throughout my career
we needed right now eleven billion accountants
I did find jobs.Went for accounting courses
difficulty had had started after college
They also required to live in dorms
I do take comfort in how far I came
they had money from rich people
they had money Because private
my housing needs were so met.I
they give just extra money
next 4 years were So tough
I maybe taken bigger risks
I was that good at my jobs
They also required met.I
me get job in government
college was in Midwest
I probably qualified for so much assistance
though my budget.I live in home in area
I just ground through it.The
I live far as how I bridged
I was working at_time 4 AM
I could go from 1 to 8
I 'll go again hungry
I graduated with two undergrad degrees
I 'm first in my family go to college
I 'm in my second year of law school
He working decent job with benefits
I ended up janitor for long time
I took out as much financial aid
I got into very good law school
we 're first in our families
side jobs make extra money
I was aimless in school
He has great work ethic
He kept to better jobs
Im Even still called egghead by my family
life is like like most Americans
my family Most of been has jail
I should have life by accounts
I do so in nicest way possible
they spew at times
I work wise
it So 's relying on my King of procrastination father
I So dropped at_time about 4 months from graduation
I 've had money since Ever first golden semester
I could just get my financial aid information
Financial Aid Loans cover my living expenses
my grandparents are living on retirement
my family was because verbally abusive
It 's taken toll so far on my grades
information was in because in time
I remind at_time months in advance
I 've had at least 50 % less money
I wanted to do more with my life
I working at least 30 hours week
I ended up at_time last semester
all was well in because in time
I 've had go to school meaning
my grades continue to suffer
it still took about 3 weeks
I could just get in in time
I file for next year 's aid
I filed at_time first time
I started in January 2010
I just focus on school
I working to maintain
everything was great
I went back home to
I got into college
I finish my degree
I moved away home
It has taken far
I So got my GED
I found job
I ended having to withdraw for certain semesters
I going on and off since I graduated in 2009
they were reducing my per term financial aid
I 'll put so hopefully into bullet points
I doing well in their classroom structure
I started at local community college
I went to various four years schools
few times took decent period of time
you do document relationship ending
school mentioned in previous bullet
I take time off before I started
I attempted in Once good spot
I went between between times
I go back at_time last year
they were reducing to point
I graduated school in 2009
I 've had Since first time
school essentially told me
I also had untreated ADHD
I enough money for gas
I need to pay 10 weeks
I lost at one point
I chose to go there
I 'm old 29 years
I lost my job
's has Associate of Science from North Texas community college
I actually attained my degree about semester ago
I 'm 20 year old third year college student
I graduated in from as also same city
my mom found out at_time Last year
I really feel like time has come
I 'm still living with my mother
time has come for me to take
I First 'm going to upfront
UT has expensive school
it has very difficult
it live normal life
me take next step
my dad 's shop started until until latter part of my second year
I got decent paying job netting about 20 $ hour
I felt like my life was finally going somewhere
I used to move into apartment on school campus
I feel for first time actual drive to finish
My last semester stopped going altogether
her have decent stable jobs in new town
I done wrong compiled into my now life
I had just gotten out relationship
I really need advice from anyone
I working at my dad 's business
I went into slight depression
I was pushed to do in school
dead end job is in warehouse
I would only go on test days
I started skipping to point
she become much more caring
I was at During time school
me have stable jobs in town
I lived in freezing winter
I lived for almost 2 weeks
I lived before my dad let
I found something in life
I ended with sub 2.0 GPA
I making sort of money
I lost job because it
me feel at_time time
debt is with degree
I have friends
I do good
we screwed from over financially my stepdad 's misuse of our money
my step father attempted at_time suicide multiple times
I was already struggling as last few years of my life
I get airline ticket so as prize in lottery at work
I So booked trip right before fall semester started
our relatives kept as much at_time next day
I have attended three semesters of college
I So went to my former childhood therapist
I have attended as degree seeking student
I went to counseling center at university
life got again at_time even more so time
he So basically threatened to kill
I So emailed my academic counselor
we went to school full of stress
next semester would would great
he Finally was told to move out
he So basically threatened kill
I so continued my student job
us be good christian children
I called financial aid center
my Mom started Around time
he move after hospitalized
I went after breaking down
our relatives is in town
I thought for odd reason
I So emailed about it
money go to college
We go on boat ride
me was just lazy
I told myself
I had worked lot before things went south for family
my parents have struggled at_time little background
it has where has incredibly difficult
** is most stressful thing in my life
I still had good amount of savings
I think been has incredibly tough
I also worked lot through college
breaking point is with my family
I bought at_time car junior year
my life work out financial hole
my parents are no longer able
I did twice without thinking
Ive made financial mistakes
I cut off financially them
I just think his industry
I call from family member
I took job After college
I took job in new city
** is thing in my life
lot is in high school
My family needed help
I make tough decision
my parents help him
Its gotten to point
me pay for college
I gave During time
I need At point
it Is too harsh
him keep job
we work hard like everyone else for things
that can help people from homeless sure
You send out resumes for teaching jobs
I had good chance of leaving here
I am so so tired of life of world
my years is in Elementary school
I was Because nerd into studying
You lose your place of living
here 's little about my life
everything want unless rich
people are where so mean
I just 'd like to know
I was without friends
few pills ending it
life has purpose
I need answers
my community college build experience remaining tuition expenses
I was able After graduating from community with associate
I got work study job at my community college to build
my bachelors tuition is in cash
grants cover community college
Used Pell grants need grants
I get entry level IT job
We have also had because is mold all over roof of house itself
I had For about a year after moved in with my Dad
him forcing to move back upstairs into heat once
more food stamps go to private charity
My Dad refused to go to Welfare office
I eventually moved into basement room
I am able for future academic years
I just moved all of my Dad 's stuff
I finally convinced After month
level went down After summer
I am moving in few days
My parents are divorced
I just moved out it
him forcing to move
me move into heat
I also got job
she sat In keeping with appearance of good mother
I was living in cheaper of two motels in town
My English teacher even provided with mantra
I had chosen 3 hour drive from where lived
her four children is in currently school
month so start looking for part time job
my home life would fade into background
I tried to speak with people at college
I ended up working overnights at Target
I was living in cheaper of two motels
mornings would finish to get to class
I tried speak with people at college
My school came through again for me
tax forms prove my parents ' income
she just showed up at_time one day
I had found house By end of summer
needed understood my situation
enough money even try to save
school was supposed to start
My teachers were my friends
I would finish work in time
I had always had my school
My grades so did my future
little girl being told she
I made pretty good grades
I so moved month so early
I did still live at home
me stay home from school
I graduated high school
my counselor So told me
Finding was harder
She would tell me
I saved money
I need help
$ 11,000 was used past year with grants free aid covering besides
my EFC went up ton just because my sister moved out
college attended for 2017 2018 school year cost
I just finished my freshman year of college
my degree require to attend for five years
my degree require attend for five years
$ 22,000 year is with room
financial status is in it
I come from family
he lost in 2009
it 's just frustrating majority of other students I interact with never talk about struggling fact
it 's frustrating majority of other students I interact with never talk about struggling fact
statistics published by university
I so rely almost entirely on loans
parents is with advanced degrees
Neither really have money
it has just frustrating
I 'm in my junior year
I majority of students
I go to show
work is in senior management executive capacity
it also seems to follow into your career life
I 've dealt While 've relatively good
I 'm also very humble individual so
they hearing for past 20 odd years
my company does lot of recruiting
I work in industry
I felt so alone especially since I moved
socioeconomic differences made it hard
it hard for me to connect to others
I had depression at_time whole time
school was predominately affluent
I especially moved away home
I felt definitely too much
I moved go to school
me make friends
Their parents push towards educational endeavors
Their parents talk how to work in their favor
Their parents talk about how world works
some just have less awareness of it
Their parents encourage to explore
They seem so much from outsider
you start if my parents had
kids entering into college
I 'm going sit here
my experience come from low income family of substance abusers
Intelligence extends far outside academic institution
I overcame Imposter Syndrome With help of my husband
I come from culture of later generation students
I wrote recently in first generation students
me feel like I did not belong in school
My GPA was at time less than 2.3
I started my bachelor 's degree
I was working full time trying
I graduating at_time next week
you push through anything life
I go to college.My background
I felt at_time depressed year
my mother preached education
My mom help out high school
my bachelor ran out money
I walking at_time time
bachelor has degree
She always told me
My parents could no longer afford to keep
money much less live if to even attend
enough savings drop down to part time
myself take large amounts of caffeine
you 'll just kill recently yourself.I
I so 've had job since 20
I did while while school
money live if to attend
I did for 3 years
awake take exam
awake go do
My boss made arrangements with my schedule to allow for full time work
I worked time because my university made on my transcript
My best friend showed up to award ceremony
I made at_time my first year as Xray tech
family friend paid for my CNA certificate
couches Various friends listening ear for
pre-requisite classes done in 1.5 years
My friends encouraging with achievement
I was expendable Before graduate school
My friends were genuinely happy for me
It still feels like I 'm stealing time
respected doctor took under his wing
I was capable based on they thought
I studied during my breaks at work
I got letter because I took while
my university refused to correct
you did well in spelling bee
I missed out on grant money
doctor took under his wing
I scraped through college
I get ready for PA school
I changed schools to get
I was when 4 years old
me work out by myself
We lived for while
I worked full time
me make about work
they thought that
I changed 4 times
I took 24 credits
I wanted in life
I was from time to time sent to other relatives
I 've 've homeless for periods of time as kid
You already live in shaky unstable world
I 've started in last couple of years
I just fucking grinded off my face
I got post degree qualification
I went to 17 different schools
I tried really hard in school
My mum dropped out school
I went common occurrence
I 'm to day 27 years now
I trashed my social life
My mum had at 15 years
you can start working
I got job in my field
We moved around lot
I moved into flat
I sacrificed lot
I saved as much
"""



listTriples = text.split("\n")

listWords = ["school", "go", "job", "year", "work", "just", "time",
             "money", "take", "lot", "life", "know", "help", "college",
             "student", "family", "also", "think", "semester", "good",
             "tell", "graduate", "end", "friend", "point", "well", "live",
             "move", "need", "hard"]


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def fineTuneModle():
    '''
    #we form the tokens using the listWords
    tokens = []
    for words in listWords:
        tkn = tokenizer(words , return_tensors="pt")
        tokens.append(tkn)

    #arguments for the model

    arguments = TrainingArguments(
        output_dir="./bert-finetuned" ,
        per_device_train_batch_size=32 ,
        num_train_epochs=3 ,
        evaluation_strategy="steps"
    )

    #bert model
    trainer = Trainer(
        args=arguments,
        model = model,
        train_dataset=tokens
    )
    '''



def generateEmbeddings(listData):
# Generate BERT embeddings for each word
    embeddings = []

    for word in listData:
        inputs = tokenizer(word, return_tensors='pt')
        outputs = model(**inputs)
        word_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        embeddings.append(word_embedding)
    return embeddings

def kMeansClustering(listData):

    # Number of clusters (you can adjust this)
    num_clusters = 10
    embeddings = generateEmbeddings(listData)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(embeddings)

    return cluster_labels


def HDBSCAN(listData):
    listData = generateEmbeddings(listData)
    model = hdbscan.HDBSCAN (min_cluster_size=5,min_samples=1)  # Adjust parameters as needed
    clustering_result = model.fit_predict(listData)

    return clustering_result


def createDictionary(cluster_labels,listData):
    dict = defaultdict(list)
    # Print the words and their assigned clusters
    for word, cluster_label in zip(listData, cluster_labels):
         dict[cluster_label].append(word)
    return dict

def printResults(dict):
    for keys in dict.keys():
        print(keys , " : " , dict[keys] , "\n\n")




from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

def sentimentAnalysis(data):
    # Load pre-trained RoBERTa model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Initialize a list to store sentiment labels
    sentiments = []

    for text in data:
        # Tokenize and format the input text
        encoded_inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        # Make a prediction using the model
        with torch.no_grad():
            output = model(**encoded_inputs)

        # Extract the predicted sentiment label
        predicted_label = output.logits.argmax(dim=1).item()
        sentiments.append(predicted_label)

    return sentiments



def createDictSentimentAnalyzed(dict):

    listDictSentiments = []

    for val in dict.values():
        data = list(val)

        #We get the sentiments for the specific cluster
        labels = sentimentAnalysis(data)

        dictLabeled = createDictionary(labels,data)
        printResults(dictLabeled)

        #return a divtionary
        listDictSentiments.append(dictLabeled)

    return listDictSentiments


labels = kMeansClustering(listTriples)
data = createDictionary(labels,listTriples)

createDictSentimentAnalyzed(data)


