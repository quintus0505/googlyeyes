The HOW-WE-TYPE-MOBILE dataset
===================================
https://userinterfaces.aalto.fi/how-we-type-m

This is the HOW-WE-TYPE-MOBILE dataset. 
It contains typing data of 30 participants typing regular finnish sentences. 
More details about the study and its procedure can be found in the paper:

Xinhui Jiang, Yang Li, Jussi P.P. Jokinen, Viet Ba Hirvola, Antti Oulasvirta, Xiangshi Ren
How We Type: Eye and Finger Movement Strategies in Mobile Typing
In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, ACM, 2020

NOTE: This subset contains only the *Gaze data* of the one-finger and two-finger condition.
Please contact jussi.jokinen@aalto.fi for more details.

----------------------------------
LICENSE AND ATTRIBUTION
----------------------------------

You are free to use this data for non-commercial use in your own research with attribution to the authors. 

Please cite: 

Xinhui Jiang, Yang Li, Jussi P.P. Jokinen, Viet Ba Hirvola, Antti Oulasvirta, Xiangshi Ren
How We Type: Eye and Finger Movement Strategies in Mobile Typing
In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, ACM, 2020

@inproceedings{Jiang2020,
author = {Jiang, Xinhui and Li, Yang, and Jokinen, Jussi P P and Hirvola, Viet Ba and Oulasvirta, Antti and Ren, Xiangshi},
booktitle = {Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems (CHI '20)},
title = {How We Type: Eye and Finger Movement Strategies in Mobile Typing.},
year = {2020}
}

----------------------------------
CONTENT
----------------------------------
  
- Gaze:
  recorded at 30 fps.
  x and y position of the gaze.
  In the data, the upper-left corner of the screen is the origin (0,0), 
  with x axis values increasing toward the right of the device and y values from topto bottom. 
  explanation of data columns: see below
  
- Keyboard_coordinates.csv
  The x-y coordinates of the center of each key on the soft keyboard.
  The keyboard had a Finnish layout
  
- Background.csv
  Subjective responses to the survey, filled by the participants after the study.

- Sentences.csv
  Sentence id and sentence.
 

----------------------------------
EXPLANATION OF DATA COLUMNS
----------------------------------

Gaze data: 

	id: int, 3-digit number
	subject id.

	block: int, 1-digit number
	1 for one-finger typing, 2 for two-finger typing.

	sentence_n: int
	sentence number, sentences marked in sentence.csv

	trialtime: int
	time stamps corresponding to each sentence,
	starts from each first tap on the keyboard.

	x, y: float
	the coordinates of the gaze.
	...
	
===============================================	