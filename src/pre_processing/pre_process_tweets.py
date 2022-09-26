import regex as re
import demoji


def remove_urls(tweet):
    tweet = re.sub("([^ ]*\.com[/A-Za-z0-9]*)", "", tweet)
    tweet = re.sub(r"(?:\http?\://|https?\://|www)\S+", "", tweet)
    return tweet


def replace_urls(tweet):
    tweet = re.sub("([^ ]*\pic.twitter)\S*", "[picture]", tweet)
    tweet = re.sub("([^ ]*\.com[/A-Za-z0-9]*)", "[url]", tweet)
    tweet = re.sub(r"(?:\http?\://|https?\://|www)\S+", "[url]", tweet)
    return tweet


def remove_at(tweet):
    tweet = re.sub(r"@", "", tweet)
    return tweet


def remove_emojis(tweet):
    dem = demoji.findall(tweet)
    for item in dem.keys():
        tweet = tweet.replace(item, "")
    return tweet


def replace_emojis(tweet):
    dem = demoji.findall(tweet)
    for key, value in dem.items():
        tweet = tweet.replace(key, value+" ")
    return tweet


def remove_handle(tweet):
    hyphen_positions = []
    for i in range(len(tweet)):
        if tweet[i] == '—':
            hyphen_positions.append(i)
    if hyphen_positions:
        last_hyphen = hyphen_positions[len(hyphen_positions) - 1]
        tweet = tweet[:last_hyphen]
    return tweet


def handle_keep_only_name(tweet):
    bracket_position = []
    for i in range(len(tweet)):
        if tweet[i] == '(':
            bracket_position.append(i)
    if bracket_position:
        last_bracket = bracket_position[len(bracket_position) - 1]
        tweet = tweet[: last_bracket]
    return tweet


def handle_keep_only_date(tweet):
    hyphen_positions = []
    for i in range(len(tweet)):
        if tweet[i] == '—':
            hyphen_positions.append(i)
    if hyphen_positions:
        last_hyphen = hyphen_positions[len(hyphen_positions) - 1]
        handle = tweet[last_hyphen+1:]
        bracket_position = []
        for i in range(len(handle)):
            if handle[i] == ')':
                bracket_position.append(i)
        if bracket_position:
            last_bracket = bracket_position[len(bracket_position) - 1]
            handle = handle[last_bracket + 1:]
        tweet = tweet[: last_hyphen] + handle
    return tweet


# tweet_1 = "an adorable candid video of Steve Martin with a dachshund ❤️❤️❤️ https://t.co/dyz41okl3m — germanbini 🌹 peace, love, brotherhood (@germanbini) November 15, 2021"
# tweet_2 = "@italiaricci  You  should  get  one  for  your  house!  #PizzaVendingMachine  #NowIWantOne  pic.twitter.com/3SV5Z9bAuX  —  Stephanie  (@Xxsteff22xX)  July  1,  2015"
# tweet_3 = "John Connally's suit and bloody shirt from Dallas on display in lobby of State Archives #jfk50pic.twitter.com/gUel0oMhuy — Aman Batheja (@amanbatheja) November 20, 2013"
# tweet_4 = "Flagship Food Group Recalls Frozen Cauliflower Because of Possible Health Risk https://t.co/gq9L6YUCuS pic.twitter.com/ckPp9EuRbo — U.S. FDA Recalls (@FDArecalls) November 26, 2021"



