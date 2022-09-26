
# Evaluation of Pre-Processing 

##Pre-Processing Tweets:

**1. URLs:**\
1A: do nothing\
1B: remove\
1C: replace with tokens ([picture] or [weblink])

**2. @:**\
2A: do nothing\
2B: remove

**3. Emojis**\
3A: do nothing\
3B: remove\
3C: replace with description of emoji

**4. Twitter Handle**\
4A: do nothing\
4B: remove\
4C: keep name\
4D: keep date

###Evaluation for dataset clef_2022_checkthat_2a_english

|URLs|@|Emojis|Twitter Handle|Map@5|
|----|---|---|---|---|
|1A|2A|3A|4A|0.9337|
|**1B**|2A|3A|4A|**0.9342**|
|**1C**|2A|3A|4A|0.9310|
|1A|**2B**|3A|4A|0.9336|
|1A|2A|**3B**|4A|0.9337|
|1A|2A|**3C**|4A|**0.9361**|
|1A|2A|3A|**4B**|0.9162|
|1A|2A|3A|**4C**|0.9272|
|1A|2A|3A|**4D**|0.9208|
|**1B**|2A|**3C**|4A|**0.9366**|

--> remove URLs and replace Emojis with descriptions

###Evaluation for dataset clef_2021_checkthat_2a_english

|URLs|@|Emojis|Twitter Handle|Map@5|
|----|---|---|---|---|
|1A|2A|3A|4A|0.9035
|**1B**|2A|3A|4A|**0.9064**|
|1A|2A|**3C**|4A|**0.9068**|
|**1B**|2A|**3C**|4A|**0.9105**|

--> remove URLs and replace Emojis with descriptions

###Evaluation for dataset clef_2020_checkthat_2_english

|URLs|@|Emojis|Twitter Handle|Map@5|
|----|---|---|---|---|
|1A|2A|3A|4A|0.9525|
|**1B**|2A|3A|4A|**0.9537**|
|1A|2A|**3C**|4A|0.9525|
|**1B**|2A|**3C**|4A|**0.9537**|

--> remove URLS










