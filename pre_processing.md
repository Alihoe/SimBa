
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
|**1A**|2A|3A|4A|0.9337|
|**1B**|2A|3A|4A| |
|**1C**|2A|3A|4A| |
|1A|**2A**|3A|4A| |
|1A|**2B**|3A|4A| |
|1A|2A|**3A**|4A| |
|1A|2A|**3B**|4A| |
|1A|2A|**3C**|4A| |
|1A|2A|3A|**4A**| |
|1A|2A|3A|**4B**| |
|1A|2A|3A|**4C**| |
|1A|2A|3A|**4D**| |










