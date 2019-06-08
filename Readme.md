Firstly save the indic.py file in the folder of the program that will call this module

Secondly import the module in the programm "import indic.py as ind"

Thirdly to do sentiment analysis on 1)Hindi text do :  model,tokenizer,len=ind.hin_model()
						     senti,acc=ind.sentiment(model,tokenizer,len,text)

						     Where: text is the hindi text on which you want to detect the polarity
							    senti stores the polarity of text
							    acc stores the accuracy

				   2)Assamese Text do:	 model,tokenizer,len=ind.asm_model()
						     	senti,acc=ind.sentiment(model,tokenizer,len,text)

						     	Where: text is the hindi text on which you want to detect the polarity
							       senti stores the polarity of text
							        acc stores the accuracy	