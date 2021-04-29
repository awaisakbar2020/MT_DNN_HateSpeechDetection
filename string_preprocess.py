import pandas as pd
import re

class string_Preprocess:
    def __init__(self):
        return;

    def remove_special_chars(self,sentence,dataset_name="twitter"):
       # print("self:", self)
        #print("sentence:", sentence)
        if dataset_name == "twitter":
            sentence = self.remove_twitter_special_chars(sentence);
        elif dataset_name == "wikipedia":
            sentence = self.remove_wikipedia_special_chars(sentence);

        sentence = sentence.lower()
        sentence = re.sub('\\n', ' ', sentence).strip()
        sentence = re.sub('http[s]?://\S+', ' ', sentence) #remove hyperlinks

        ## process string
        sentence = re.sub('(?<=\d)[,.](?=\d)','',sentence)
        sentence = sentence.replace("="," ")
        sentence = sentence.replace("&"," ")
        sentence = sentence.replace("|"," ")
        sentence = sentence.replace(";"," ")
        sentence = sentence.replace("{","")
        sentence = sentence.replace("}","")
        sentence = sentence.replace("~","")
        sentence = sentence.replace('"'," ")
        sentence = sentence.replace('',"")
        sentence = sentence.replace("(","")
        sentence = sentence.replace(")","")
        sentence = sentence.replace("`","")
        sentence = sentence.replace("*","")
        sentence = sentence.replace("[","")
        sentence = sentence.replace("]","")
        sentence = sentence.replace(":","")
        sentence = sentence.replace('0',' ')
        sentence = sentence.replace('1',' ')
        sentence = sentence.replace('2',' ')
        sentence = sentence.replace('3',' ')
        sentence = sentence.replace('4',' ')
        sentence = sentence.replace('5',' ')
        sentence = sentence.replace('6',' ')
        sentence = sentence.replace('7',' ')
        sentence = sentence.replace('8',' ')
        sentence = sentence.replace('9',' ')
        sentence = sentence.replace('\"',"")
        sentence = sentence.replace('_'," ")
        sentence = sentence.replace('#'," ")
        
        ##Retention chars
        sentence = sentence.replace("-"," - ")
        sentence = sentence.replace("!"," ! ")
        sentence = sentence.replace(","," , ")
        sentence = sentence.replace("."," . ")

        sentence = ' '.join(sentence.split())
        ## collapse repetition
        while (True):
            insentence = sentence
            sentence = re.sub('\.\s*\.+', ' . ', sentence)
            sentence = re.sub('\,\s*\,+', ' , ', sentence)
            sentence = re.sub('(\!\s*\!)+', ' ! ', sentence)

            if insentence == sentence:
                break

        sentence = ' '.join(sentence.split())
        
        if dataset_name == "twitter":
            sentence = self.remove_twitter_special_chars(sentence);
        elif dataset_name == "wikipedia":
            sentence = self.remove_wikipedia_special_chars(sentence);

        return sentence;

    def remove_twitter_special_chars(self,sentence):
        sentence = re.sub(" RT "," ",sentence)
        sentence = re.sub(r'^RT\W'," ",sentence)
        sentence = re.sub(r'\WRT\W'," ",sentence)
        sentence = re.sub(r'\WRT$'," ",sentence)
        sentece = re.sub(r'^[^A-Za-z]*RT', ' ', sentence )
        #sentence = re.sub(r'[@]\w+ ?', ' ', sentence).strip()
        sentence = re.sub(r'[@]\w+ ?', ' @USER ', sentence).strip()

        ## Get Hashtag

        match = re.findall(r'\#([A-Za-z]*)',sentence)
        #print("Debug: Match:", match)
        for completeTag in  match:
            match2 = re.findall(r'([A-Za-z]?[a-z]*)',completeTag)
            words_in_tag=""
            #print("Debug: Match2:", match2)
            
            for word in match2:
                words_in_tag+= word+" "
            words_in_tag=words_in_tag.strip()
            #print("Debug: words:", words_in_tag)
            #print("Debug: completeTag:", completeTag)
            sentence = re.sub(completeTag, words_in_tag, sentence )
        return sentence;

    def remove_wikipedia_special_chars(self,sentence):
        sentence = sentence.replace("NEWLINE_TOKEN"," ")
        sentence = sentence.replace("== Warning =="," ")
        return sentence;

